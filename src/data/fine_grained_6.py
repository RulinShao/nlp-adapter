import collections
import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from args import args


DATASETS=('cubs_cropped', 'stanford_cars_cropped', 'flowers', 'wikiart', 'sketches')
NUM_CLASSES=(200, 196, 102, 195, 250)
INIT_LR=(1e-3, 1e-2, 1e-3, 1e-3, 1e-3)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
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
                                 transforms.Resize(256),
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
        transforms.Resize(224),
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
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""

    args.train_path = '~/dataset/fine_grained/%s/train' % (args.dataset)

    if (args.dataset in ['imagenet', 'face_verification', 'emotion', 'gender'] or
        args.dataset[:3] == 'age'):
        args.val_path = '~/dataset/fine_grained/%s/val' % (args.dataset)
    else:
        args.val_path = '~/dataset/fine_grained/%s/test' % (args.dataset)


def get_loaders(args):
    set_dataset_paths(args)
    if 'cropped' in args.dataset:
        t_loader = train_loader_cropped(args.train_path, args.batch_size)
        v_loader = val_loader_cropped(args.val_path, args.val_batch_size)
    else:
        t_loader = train_loader(args.train_path, args.batch_size)
        v_loader = val_loader(args.val_path, args.val_batch_size)
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