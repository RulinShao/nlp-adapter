"""
This script is adapted from https://github.com/RAIVNLab/supsup.
"""
import os
import pathlib
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from args import args
# import adaptors
import data
import schedulers
import trainers
import utils
from models.utils import get_backbone, get_task_model, modify_model


from timm.models import load_checkpoint, create_model
import models.vit

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "settings.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    print(f"=> Saving data in {run_base_dir}")

    # Get dataloader.
    data_loader = getattr(data, args.set)()

    task_length = 1000 // args.num_tasks

    # Get the backbone model with a new head layer.
    model = get_backbone()
    model = modify_model(model, task_length)

    # Put the model on the GPU,
    model = utils.set_gpu(model)

    # Track accuracy on all tasks.
    if args.num_tasks:
        best_acc1 = [0.0 for _ in range(args.num_tasks)]
        curr_acc1 = [0.0 for _ in range(args.num_tasks)]
        adapt_acc1 = [0.0 for _ in range(args.num_tasks)]

    criterion = nn.CrossEntropyLoss().to(args.device)

    writer = SummaryWriter(log_dir=run_base_dir)

    # Track the number of tasks learned.
    num_tasks_learned = 0

    trainer = getattr(trainers, args.trainer or "default")
    print(f"=> Using trainer {trainer}")

    train, test = trainer.train, trainer.test

    # Iterate through all tasks.
    for idx in range(args.num_tasks or 0):
        print(f"Task {args.set}: {idx}")

        # Update the data loader so that it returns the data for the correct task, also done by passing the task index.
        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(idx)

        # Get the model modified for the specific task and trainable params.
        # TODO:
        #  1. return best_acc1 when resuming from a ckpt.
        #  2. load test related adapters when resume.
        model, params = get_task_model(model, num_tasks_learned, idx)

        # train_weight_tasks specifies the number of tasks that the weights are trained for.
        # e.g. in SupSup, train_weight_tasks = 0. in BatchE, train_weight_tasks = 1.
        # If training weights, use train_weight_lr. Else use lr.
        lr = (
            args.train_weight_lr
            if args.train_weight_tasks < 0
               or num_tasks_learned < args.train_weight_tasks
            else args.lr
        )

        # get optimizer, scheduler
        if args.optimizer == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=args.wd)
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(params, lr=lr)
        else:
            optimizer = optim.SGD(
                params, lr=lr, momentum=args.momentum, weight_decay=args.wd
            )

        train_epochs = args.epochs

        if args.no_scheduler:
            scheduler = None
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)

        # Train on the current task.
        for epoch in range(1, train_epochs + 1):
            train(
                model,
                writer,
                data_loader.train_loader,
                optimizer,
                criterion,
                epoch,
                idx,
                data_loader,
            )

            # Required for our PSP implementation, not used otherwise.
            # utils.cache_weights(model, num_tasks_learned + 1)

            curr_acc1[idx] = test(
                model, writer, criterion, data_loader.val_loader, epoch, idx
            )
            if curr_acc1[idx] > best_acc1[idx]:
                best_acc1[idx] = curr_acc1[idx]
            if scheduler:
                scheduler.step()

            if (
                    args.iter_lim > 0
                    and len(data_loader.train_loader) * epoch > args.iter_lim
            ):
                break

        utils.write_result_to_csv(
            name=f"{args.name}~set={args.set}~task={idx}",
            curr_acc1=curr_acc1[idx],
            best_acc1=best_acc1[idx],
            save_dir=run_base_dir,
        )

        if args.save == "full":
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / f"task{idx}_full_final.pt",
            )
        elif args.save == "adapter":
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": {k: v for k, v in model.state_dict().items()
                                  if 'bn' in k or 'adapter' in k or 'head' in k},
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / f"task{idx}_adapter_final.pt",
            )
        elif args.save == "head":
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": {k: v for k, v in model.state_dict().items()
                                   if 'head' in k},
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / f"task{idx}_adapter_final.pt",
            )

        # Save memory by deleting the optimizer and scheduler.
        del optimizer, scheduler, params

        # Increment the number of tasks learned.
        num_tasks_learned += 1

        # If operating in NNS scenario, get the number of tasks learned count from the model.
        if args.trainer and "nns" in args.trainer:
            model.apply(
                lambda m: setattr(
                    m, "num_tasks_learned", min(model.num_tasks_learned, args.num_tasks)
                )
            )
        else:
            model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks_learned))

    return adapt_acc1


# TODO: Remove this with task-eval
def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": args.wd, },
                {"params": rest_params, "weight_decay": args.wd},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


if __name__ == "__main__":
    main()