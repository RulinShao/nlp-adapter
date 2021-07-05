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

    # Get the backbone model.
    model = get_backbone()

    # Track accuracy on all tasks.
    if args.num_tasks:
        best_acc1 = [0.0 for _ in range(args.num_tasks)]
        curr_acc1 = [0.0 for _ in range(args.num_tasks)]
        adapt_acc1 = [0.0 for _ in range(args.num_tasks)]
        alpha_list = []

    criterion = nn.CrossEntropyLoss().to(args.device)

    writer = SummaryWriter(log_dir=run_base_dir)

    # Track the number of tasks learned.
    num_tasks_learned = 0

    trainer = getattr(trainers, args.trainer or "default")
    print(f"=> Using trainer {trainer}")

    train, test, infer, adapt_test = trainer.train, trainer.test, trainer.infer, trainer.adapt_test

    # Iterate through all tasks.
    for idx in range(args.num_tasks or 0):
        print(f"Task {args.set}: {idx}")

        # Update the data loader so that it returns the data for the correct task, also done by passing the task index.
        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(idx)

        task_length = data_loader.num_classes
        task_name = data_loader.dataset_name
        model = modify_model(model, task_length)

        if args.train_adapter and args.capacity is not None:
            model, params, adapter_params = get_task_model(model, num_tasks_learned, idx)
        else:
            model, params = get_task_model(model, num_tasks_learned, idx)

        if args.train_adapter and args.capacity is not None:
            infer(model, idx, writer, criterion, data_loader.train_loader, use_soft=args.soft_alpha)
            alpha_list.append(model.module.alpha.clone())

        # get learning rate
        lr = (
            args.train_weight_lr
            if args.train_weight_tasks < 0
               or num_tasks_learned < args.train_weight_tasks
            else args.lr
        )

        # get optimizer, scheduler
        if args.optimizer == "adam":
            if args.train_adapter and args.capacity is not None:
                for d in range(model.module.depth):
                    for i in range(2):
                        for c in range(model.module.capacity):
                            params_prefix = f"blocks.{d}.adapter{i+1}.{c}."
                            num_adapter_learned = model.module.adapter_count[d][i][c] if model.module.adapter_count[d][i][c] < 10 else 10
                            if d + i + c == 0:
                                optimizer = optim.Adam([
                                    {'params': adapter_params[params_prefix], 'lr': lr / 2**num_adapter_learned}
                                ])
                            else:
                                optimizer.param_groups.append({'params': adapter_params[params_prefix], 'lr': lr / 2**num_adapter_learned})
                optimizer.param_groups.append({'params': params, 'lr': lr})

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

        if args.train_adapter and args.capacity is not None:
            model.module.count_alpha()

        utils.write_result_to_csv(
            name=f"{args.name}~set={args.set}~task={task_name}",
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

                adapt_acc, alpha_diff = adapt_test(
                    model, writer, criterion, data_loader.val_loader, i, alpha_list, args.soft_alpha
                )

                adapt_acc1[i] = adapt_acc
                avg_acc += adapt_acc
                avg_bwt += (adapt_acc1[i] - curr_acc1[i])

                torch.cuda.empty_cache()

                if alpha_diff is not None:
                    writer.add_scalar(
                        f"fine-grained/capacity-alpha/{num_tasks_learned}", alpha_diff, i
                    )

            writer.add_scalar(
                "fine-grained/avg_acc", avg_acc / num_tasks_learned, num_tasks_learned
            )
            if num_tasks_learned > 1:
                writer.add_scalar(
                    "fine-grained/avg_bwt", avg_bwt / (num_tasks_learned - 1), num_tasks_learned
                )
            torch.cuda.empty_cache()

    return adapt_acc1





if __name__ == "__main__":
    main()