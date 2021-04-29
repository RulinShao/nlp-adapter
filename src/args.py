import parser as _parser

import argparse
import sys
import yaml

args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="AdapterCL")
    parser.add_argument(
        "--config", type=str, default=None, help="Config file to use, YAML format"
    )
    parser.add_argument("--name", type=str, default="100task_1full_99adapter", help="Experiment id.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/",
        help="Location to logs/checkpoints",
    )
    parser.add_argument(
        "--model_name", type=str, default="vit_small_patch16_224_adapter", help="timm model name"
    )
    parser.add_argument(
        "--pretrained", type=bool, default=False, help="whether use a pretrained model as the backbone"
    )
    parser.add_argument(
        "--train_adapter", type=bool, default=True, help="Train the adapter and norm layers only"
    )
    parser.add_argument(
        "--train_head", type=bool, default=False, help="Train the head layer only"
    )
    parser.add_argument(
        "--save", type=str, default="adapter", choices=["full", "adapter, head"], help="save full checkpoints if full, save only tranable parameter is partial"
    )
    parser.add_argument(
        "--train-weight-tasks",
        type=int,
        default=1,
        metavar="N",
        help="number of tasks to train the weights, e.g. 1 for batchensembles. -1 for all tasks",
    )
    parser.add_argument(
        "--task-eval",
        default=None,
        type=int,
        help="Only evaluate on this task (for memory efficiency and grounded task info",
    )
    parser.add_argument(
        "--multigpu",
        default="0,1,2,3",
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument("--workers", type=int, default=8, help="how many cpu workers")
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Which optimizer to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00005,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--lr_policy", type=str, default="cosine_lr", help="lr policy"
    )
    parser.add_argument(
        "--warmup_length", type=int, default=5, help="warm up length"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        metavar="M",
        help="Weight decay (default: 0.0001)",
    )

    parser.add_argument(
        "--seed", type=int, default=310, metavar="S", help="random seed (default: 310)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--data", type=str, default='/home/ec2-user/dataset/ilsvrc2012/', help="Location to store data",
    )
    parser.add_argument(
        "--num-tasks",
        default=100,
        type=int,
        help="Number of tasks, None if no adaptation is necessary",
    )
    parser.add_argument("--resume", type=str, default=None, help='optionally resume. checkpoint path')
    parser.add_argument("--model", type=str, help="Type of model.")
    parser.add_argument(
        "--eval-ckpts",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="After learning n tasks for n in eval_ckpts we perform evaluation on all tasks learned so far",
    )
    parser.add_argument("--set", type=str, default='SplitImageNet', help="Which dataset to use")
    parser.add_argument("--no-scheduler", action="store_true", help="constant LR")
    parser.add_argument(
        "--iter-lim", default=-1, type=int, help="iteration limitation"
    )
    parser.add_argument(
        "--train-weight-lr",
        default=0.1,
        type=float,
        help="While training the weights, which LR to use.",
    )
    parser.add_argument(
        "--trainer",
        default=None,
        type=str,
        help="Which trainer to use, default in trainers/default.py",
    )

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if args.config is not None:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()