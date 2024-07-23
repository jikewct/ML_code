import logging
import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from .data_trans import trans


class Cifar10:

    @staticmethod
    def get_loader(root, batch_size):

        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=trans,
            # transform=script_utils.get_transform(),
        )

        test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=trans,
            # transform=script_utils.get_transform(),
        )
        logging.info(f"train images num:{len(train_dataset)}, test images num:{len(test_dataset)}, dataset path:{root}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
        )

        return train_loader, test_loader
