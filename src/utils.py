from PIL import Image
import time
import pathlib

import torch
import torch.nn as nn
import models
import torch.backends.cudnn as cudnn

from args import args


def set_gpu(model=None):
    if args.multigpu is None:
        args.device = torch.device("cpu")
        return model
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        args.device = torch.cuda.current_device()
        cudnn.benchmark = True
        if model is not None:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
                args.multigpu[0]
            )  
            return model


def write_result_to_csv(**kwargs):
    results = pathlib.Path(args.log_dir) / "results.csv"

    if not results.exists():
        results.write_text("Date Finished,Name,Current Val,Best Val,Save Directory\n")

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{name}, "
                "{curr_acc1:.04f}, "
                "{best_acc1:.04f}, "
                "{save_dir}\n"
            ).format(now=now, **kwargs)
        )


def save_ckpt(model, best_acc1, curr_acc1, run_base_dir, idx):
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


def get_cifar_loaders(args):
    from torchvision import datasets, transforms

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        args.data_dir, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        args.data_dir, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def set_seed(args):
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def make_work_dir(args):
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
    return run_base_dir
