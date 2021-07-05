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