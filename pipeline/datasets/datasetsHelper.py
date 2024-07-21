from .cifar10 import Cifar10
from .mnist import Mnist
from .lsun_church import Lsun


def get_loader(dataset, root, batch_size, img_size = None):
    if dataset == "mnist":
        return Mnist.get_loader(root, batch_size)
    elif dataset == "cifar10":
        return Cifar10.get_loader(root, batch_size)
    elif dataset == "lsun":
        return Lsun.get_loader(root, batch_size, img_size)
    else:
        raise ValueError("unknown dataset")


def get_dataset_arguments(dataset):
    if dataset == "mnist":
        return Mnist.img_size, Mnist.img_channels
    else:
        raise ValueError("unknown dataset")
