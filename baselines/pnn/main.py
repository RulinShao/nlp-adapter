"""
Progressive Neural Networks
original paper: https://arxiv.org/abs/1606.04671
reference code: https://github.com/TomVeniat/ProgressiveNeuralNetworks.pytorch
Adapted for SplitImageNet dataset
"""

import argparse

import os

import numpy as np
import torch
import torch.nn.functional as F

import logging
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from tqdm import tqdm

# from src.data.PermutedMNIST import get_permuted_MNIST
from src.data.splitimagenet import get_SplitImageNet
from src.model.ProgressiveNeuralNetworks import PNN
from src.tools.arg_parser_actions import LengthCheckAction
from src.tools.evaluation import evaluate_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks for SplitImageNet')
    parser.add_argument('-path', default='/home/ec2-user/dataset/ilsvrc2012/', type=str, help='path to the data')
    parser.add_argument('-cuda', default=0, type=int, help='Cuda device to use (-1 for none)')
    parser.add_argument('-run_base_dir', default='outputs', type=str, help='dir to save outputs')

    parser.add_argument('--layers', metavar='L', type=int, default=3, help='Number of layers per task')
    # parser.add_argument('--sizes', dest='sizes', default=[784, 1024, 512, 10], nargs='+',
    #                     action=LengthCheckAction)
    parser.add_argument('--sizes', dest='sizes', default=[3*784, 1024, 512, 10], nargs='+',
                        action=LengthCheckAction)

    parser.add_argument('--n_tasks', dest='n_tasks', type=int, default=100)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--bs', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4, help='Optimizer weight decay')
    parser.add_argument('--momentum', dest='momentum', type=float, default=1e-4, help='Optimizer momentum')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')

    args = parser.parse_known_args()
    return args[0]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])
    writer = SummaryWriter(log_dir=args['run_base_dir'])

    model = PNN(args['layers'])

    train_loaders, val_loaders = get_SplitImageNet(args['path'], args['batch_size'], args['workers'], args['n_tasks'])

    x = torch.Tensor()
    y = torch.LongTensor()

    if args['cuda'] != -1:
        logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running WITHOUT cuda')

    for task_id, train_set in enumerate(train_loaders):
        # val_perf = evaluate_model(model, x, y, val_set, task_id=task_id)

        model.freeze_columns()
        model.new_task(args['sizes'])

        optimizer = torch.optim.RMSprop(model.parameters(task_id), lr=args['lr'],
                                        weight_decay=args['wd'], momentum=args['momentum'])

        train_accs = []
        train_losses = []
        for epoch in range(args['epochs']):
            total_samples = 0
            total_loss = 0
            correct_samples = 0
            for inputs, labels in tqdm(train_set):
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                x = x.view(x.size(0), -1)
                predictions = model(Variable(x))

                _, predicted = torch.max(predictions.data, 1)
                total_samples += y.size(0)
                correct_samples += (predicted == y).sum()

                indiv_loss = F.cross_entropy(predictions, Variable(y))
                total_loss += indiv_loss.data[0]

                optimizer.zero_grad()
                indiv_loss.backward()
                optimizer.step()

            train_accs.append(correct_samples / total_samples)
            train_losses.append(total_loss / total_samples)
            logger.info(
                '[T{}][{}/{}] Loss={}, Acc= {}'.format(task_id, epoch, args['epochs'], train_losses[-1],
                                                       train_accs[-1]))
            writer.add_scalar(
                f"train/{task_id}:acc", train_accs[epoch], epoch
            )
            writer.add_scalar(
                f"train/{task_id}:loss", train_losses[epoch], epoch
            )

        avg_pergs = 0.0
        logger.info('Evaluation after task {}:'.format(task_id))
        for i in range(task_id + 1):
            val_loader = val_loaders[i]
            test_perf = evaluate_model(model, x, y, val_loader)
            avg_pergs += test_perf
            logger.info('\tTest {}°{} - acc:{}%'.format(i, task_id, test_perf))
            writer.add_scalar(
                f"test_all(%)/{task_id}", test_perf, i
            )

        writer.add_scalar(
            f"test_avg(%)", avg_pergs/(task_id+1), task_id
        )


if __name__ == '__main__':
    main(vars(get_args()))
