import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

import trainers
import utils
from models.utils import get_backbone, get_task_model, modify_model

from timm.models import load_checkpoint, create_model
import models.avit
from models.avit import vit_base_patch16_224_in21k_adapter


import argparse


def main():
    parser = argparse.ArgumentParser(description="AdapterCL")

    parser.add_argument("--name", type=str, default="730:in21k->cifar10", help="Experiment id.")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224_in21k_adapter", help="timm model name")
    parser.add_argument("--pretrained", type=bool, default=True, help="whether use a pretrained model as the backbone")
    parser.add_argument("--train_adapter", type=bool, default=True, help="Train the adapter and norm layers only")
    parser.add_argument("--capacity", type=int, default=None,
                        help="The maximum of the number of adapters can be added.")
    parser.add_argument("--soft_alpha", type=bool, default=False)
    parser.add_argument("--train_layer", type=int, default=-1,
                        help="Train the last n layers. 0 for the head layer only.")

    parser.add_argument("--multigpu", default="0,1,2,3", type=lambda x: [int(a) for a in x.split(",")],
                        help="Which GPUs to use for multigpu training", )
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)", )
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)", )
    parser.add_argument("--iter-lim", default=-1, type=int)
    parser.add_argument("--workers", type=int, default=8, help="how many cpu workers")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use")
    parser.add_argument("--lr", type=float, default=0.00005, metavar="LR",
                        help="learning rate (default: 0.1, 0.00005, 0.0005)", )
    parser.add_argument("--lr_policy", type=str, default="exp_lr", choices=["cosine_lr", "exp_lr"], help="lr policy")
    parser.add_argument("--warmup_length", type=int, default=5, help="warm up length")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="Momentum (default: 0.9)", )
    parser.add_argument("--wd", type=float, default=0.000, help="Weight decay (default: 0.0001)", )
    parser.add_argument("--train-weight-lr", default=0.00005, type=float,
                        help="While training the weights, which LR to use.", )
    parser.add_argument("--seed", type=int, default=310, metavar="S", help="random seed (default: 310)")

    parser.add_argument("--log-dir", default="outputs/transfer/")
    parser.add_argument("--log-interval", default=1, type=int)
    parser.add_argument("--save", type=str, default="full", choices=["full", "adapter", "head", "layer"],
                        help="save full checkpoints if full, save only tranable parameter is partial")

    parser.add_argument("--num-class", default=10, type=int)
    parser.add_argument("--data-dir", default="../../dataset/")
    parser.add_argument("--img-size", default=224, type=int)

    args = parser.parse_args()


    utils.set_seed(args)

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    run_base_dir = utils.make_work_dir(args)

    # Get dataloader.
    train_loader, test_loader = utils.get_cifar_loaders(args)

    # Track accuracy.
    best_acc1 = 0.0
    curr_acc1 = 0.0

    writer = SummaryWriter(log_dir=run_base_dir)

    trainer = getattr(trainers, "default")
    print(f"=> Using trainer {trainer}")

    train, test= trainer.train, trainer.test

    task_length = args.num_class

    # Get the backbone model.
    # model = get_backbone(img_size=args.img_size)
    model = eval(args.model_name)(
        pretrained=args.pretrained,
        img_size=args.img_size, num_classes=args.num_class)
    # , patch_size=args.patch)
    model = modify_model(model, task_length)
    model = utils.set_gpu(model, args)

    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.train_adapter and args.capacity is not None:
        model, params, adapter_params = get_task_model(model, 0, 0)
    else:
        model, params = get_task_model(model, 0, 0)

    # get learning rate
    lr = args.lr

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

    if args.lr_policy == "cosine_lr":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)
    elif args.lr_policy == "exp_lr":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    else:
        assert not args.no_scheduler; "wrong lr scheduler!"
        scheduler = None

    # Train on the current task.
    for epoch in range(1, train_epochs + 1):
        train(
            model,
            writer,
            train_loader,
            optimizer,
            criterion,
            epoch,
            0,
            args=args
        )

        curr_acc1 = test(
            model, writer, criterion, test_loader, epoch, 0, args=args
        )
        if curr_acc1 > best_acc1:
            best_acc1 = curr_acc1
        if scheduler:
            scheduler.step()

        if (
                args.iter_lim > 0
                and len(train_loader) * epoch > args.iter_lim
        ):
            break

    if args.train_adapter and args.capacity is not None:
        model.module.count_alpha()

    writer.add_scalar(
        f"cifar{args.num_class}/best_acc", best_acc1
    )

    utils.write_result_to_csv(
        name=f"{args.name}~set=cifar{args.num_class}",
        curr_acc1=curr_acc1,
        best_acc1=best_acc1,
        save_dir=run_base_dir,
        args=args,
    )

    utils.save_ckpt(model, best_acc1, curr_acc1, run_base_dir, 0, args)

    # Save memory by deleting the optimizer and scheduler.
    del optimizer, scheduler, params, model



if __name__ == "__main__":
    main()