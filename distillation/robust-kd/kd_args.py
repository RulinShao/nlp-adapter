import argparse


args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="KD")
    parser.add_argument(
        "--teacher_model", type=str, default="vit_small_patch16_224_adapter", help="teacher model name"
    )
    parser.add_argument(
        "--student_model", type=str, default="resnet18", help="student model name"
    )
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
        default=0.0005,
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
    parser.add_argument("--workers", type=int, default=4, help="how many cpu workers")
    parser.add_argument("--name", type=str, default="default", help="Experiment id.")
    parser.add_argument(
        "--datasets", type=str, default='../../../dataset/', help="Location to store datasets",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="ouputs",
        help="Location to logs/checkpoints",
    )
    parser.add_argument("--resume", type=str, default=None, help='optionally resume')
    parser.add_argument("--model", type=str, help="Type of model.")
    parser.add_argument(
        "--multigpu",
        default="0,1",
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--eval-ckpts",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="After learning n tasks for n in eval_ckpts we perform evaluation on all tasks learned so far",
    )
    parser.add_argument(
        "--num-tasks",
        default=10,
        type=int,
        help="Number of tasks, None if no adaptation is necessary",
    )
    parser.add_argument("--set", type=str, default='ImageNet', help="Which dataset to use")
    parser.add_argument(
        "--save", action="store_true", default=True, help="save checkpoints"
    )
    parser.add_argument("--no-scheduler", action="store_true", help="constant LR")
    parser.add_argument(
        "--iter-lim", default=-1, type=int, help="iteration limitation"
    )
    parser.add_argument(
        "--task-eval",
        default=0,
        type=int,
        help="Only evaluate on this task (for memory efficiency and grounded task info",
    )
    parser.add_argument(
        "--train-weight-tasks",
        type=int,
        default=0,
        metavar="N",
        help="number of tasks to train the weights, e.g. 1 for batchensembles. -1 for all tasks",
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

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()