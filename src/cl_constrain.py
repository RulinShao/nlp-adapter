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
import data as data_
import trainers
import utils
from models.utils import get_backbone, get_task_model, modify_model


from timm.models import load_checkpoint, create_model
import models.avit


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
    data_loader = getattr(data_, args.set)()

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

    train, test, infer = trainer.train, trainer.test, trainer.infer

    # Iterate through all tasks.
    for idx in range(args.num_tasks or 0):
        print(f"Task {args.set}: {idx}")

        # Update the data loader so that it returns the data for the correct task, also done by passing the task index.
        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(idx)
        model, params = get_task_model(model, num_tasks_learned, idx)

        if args.train_adapter and args.capacity is not None:
            infer(model, writer, criterion, data_loader.train_loader)

        # get learning rate
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

        utils.save_ckpt(model, best_acc1, curr_acc1, run_base_dir, idx)

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

        # Evaluate the performance on prior tasks
        # if num_tasks_learned in args.eval_ckpts or num_tasks_learned == args.num_tasks:
        if num_tasks_learned % args.eval_interval == 0:
            avg_acc = 0.0
            avg_bwt = 0.0

            # Settting task to -1 tells the model to infer task identity instead of being given the task.
            model.apply(lambda m: setattr(m, "task", -1))

            for i in range(num_tasks_learned):
                print(f"Testing {i}: {args.set} ({i})")
                model.apply(lambda m: setattr(m, "task", i))

                # Update the data loader so it is returning data for the right task.
                data_loader.update_task(i)

                # Clear the stored information -- memory leak happens if not.
                for p in model.parameters():
                    p.grad = None

                for b in model.buffers():
                    b.grad = None

                torch.cuda.empty_cache()

                adapt_acc = test(
                    model, writer, criterion, data_loader.val_loader, None, i
                )

                adapt_acc1[i] = adapt_acc
                avg_acc += adapt_acc
                avg_bwt += (adapt_acc1[i] - curr_acc1[i])

                torch.cuda.empty_cache()

            writer.add_scalar(
                "cl/avg_acc", avg_acc / num_tasks_learned, num_tasks_learned
            )
            writer.add_scalar(
                "cl/avg_bwt", avg_bwt / (num_tasks_learned - 1), num_tasks_learned
            )
            torch.cuda.empty_cache()

    return adapt_acc1





if __name__ == "__main__":
    main()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # # Load pretrained model
    # model = create_model(args.model_name,
    #                      pretrained=True,
    #                      num_classes=1000,
    #                      in_chans=3, )
    # model.eval().to(device)
    # print(dict(model.named_parameters()).keys())
