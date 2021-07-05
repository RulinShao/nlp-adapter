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