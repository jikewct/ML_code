import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from .data_trans import resize_trans


class CelebA(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        return (["dummy"], {"dummy": 0})

    @staticmethod
    def make_dataset(directory, class_to_idx, extensions, is_valid_file):
        instances = []
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, 0
                instances.append(item)
        return instances

    @staticmethod
    def get_loader(root, batch_size, img_size):

        train_dataset = CelebA(
            root=root,
            transform=resize_trans(img_size),
            # transform=script_utils.get_transform(),
        )

        val_dataset = CelebA(
            root=root,
            transform=resize_trans(img_size),
            # transform=script_utils.get_transform(),
        )
        logging.info(f"train images num:{len(train_dataset)}, val images num:{len(val_dataset)}, dataset path:{root}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=True,
        )

        return train_loader, val_loader
