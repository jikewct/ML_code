from .cifar10 import Cifar10
from .mnist import Mnist
from .lsun_church import Lsun
from .celeba import CelebA
from .afhq import Afhq


def get_loader(dataset, root, batch_size, img_size = None, img_class="all"):
    if dataset == "mnist":
        return Mnist.get_loader(root, batch_size)
    elif dataset == "cifar10":
        return Cifar10.get_loader(root, batch_size)
    elif dataset == "lsun":
        return Lsun.get_loader(root, batch_size, img_size)
    elif dataset == "celeba":
        return CelebA.get_loader(root, batch_size, img_size)
    elif dataset == "afhq":
        return Afhq.get_loader(root, batch_size, img_size, img_class)
    else:
        raise ValueError("unknown dataset")


def get_dataset_arguments(dataset):
    if dataset == "mnist":
        return Mnist.img_size, Mnist.img_channels
    else:
        raise ValueError("unknown dataset")
