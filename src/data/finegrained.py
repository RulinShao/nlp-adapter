import collections
import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from args import args

import os

import torch
from torchvision import datasets, transforms
from args import args

import torch.multiprocessing

import numpy as np

from copy import copy, deepcopy
from itertools import chain

torch.multiprocessing.set_sharing_strategy("file_system")


DATASETS=('cubs_cropped', 'stanford_cars_cropped', 'flowers', 'wikiart', 'sketches')
NUM_CLASSES=(200, 196, 102, 195, 250)
INIT_LR=(1e-3, 1e-2, 1e-3, 1e-3, 1e-3)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FineGrained:
    def __init__(self):
        super(FineGrained, self).__init__()

        data_root = os.path.join('~/dataset/fine_grained/', '')

        use_cuda = torch.cuda.is_available()

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.train_loaders = []
        self.val_loaders = []

        for dataset_name in DATASETS:
            traindir, valdir = set_dataset_paths(dataset_name)
            train_loader, val_loader = get_loaders(dataset_name, traindir, valdir, args)
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
    
    def update_task(self, i):
        self.train_loader = self.train_loaders[i]
        self.val_loader =  self.val_loaders[i]
        self.dataset_name = DATASETS[i]
        self.num_classes = NUM_CLASSES[i]
        

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def train_loader(path, train_batch_size, num_workers=24, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.ImageFolder(path, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader(path, val_batch_size, num_workers=24, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dataset = \
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def train_loader_cropped(path, train_batch_size, num_workers=24, pin_memory=False):
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.ImageFolder(path, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader_cropped(path, val_batch_size, num_workers=24, pin_memory=False):
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dataset = \
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def set_dataset_paths(dataset):
    """Set default train and test path if not provided as input."""

    train_path = '~/dataset/fine_grained/%s/train' % (dataset)

    if (dataset in ['imagenet', 'face_verification', 'emotion', 'gender'] or
        dataset[:3] == 'age'):
        val_path = '~/dataset/fine_grained/%s/val' % (dataset)
    else:
        val_path = '~/dataset/fine_grained/%s/test' % (dataset)
    
    return train_path, val_path


def get_loaders(dataset, train_path, val_path, args):
    # set_dataset_paths(args)
    if 'cropped' in dataset:
        t_loader = train_loader_cropped(train_path, args.batch_size)
        v_loader = val_loader_cropped(val_path, args.batch_size)
    else:
        t_loader = train_loader(train_path, args.batch_size)
        v_loader = val_loader(val_path, args.batch_size)
    return t_loader, v_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Data Loader")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--val_batch_size", default=32)
    args = parser.parse_args()

    for dataset_name in DATASETS:
        args.dataset = dataset_name
        t_loader, v_loader = get_loaders(args)
        print(f"{args.dataset}: {len(t_loader)}, {len(v_loader)}")