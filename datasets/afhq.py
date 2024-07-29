import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from . import dataset_factory
from .base_dataset import BaseDataset


@dataset_factory.register_dataset(name="afhq")
class Afhq(BaseDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def init_dataset(self, **kargs):
        img_class = kargs["afhq"].img_class
        if img_class == "all":
            self.train_dataset = AFHQ(
                root=self.root_path + "/train/",
                transform=self.resize_trans(self.img_size),
                img_class=img_class,
            )
            self.val_dataset = AFHQ(
                root=self.root_path + "/val/",
                transform=self.resize_trans(self.img_size),
                img_class=img_class,
            )
        else:
            self.train_dataset = AFHQ(
                root=self.root_path + "/train/",
                transform=self.resize_trans(self.img_size),
                img_class=img_class,
                # transform=script_utils.get_transform(),
            )

            self.val_dataset = AFHQ(
                root=self.root_path + "/val/",
                transform=self.resize_trans(self.img_size),
                img_class=img_class,
                # transform=script_utils.get_transform(),
            )


class AFHQ(datasets.ImageFolder):
    def __init__(self, root, transform, img_class="all"):
        self.img_class = img_class
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        if self.img_class == "all":
            return super().find_classes(directory)
        return ([self.img_class], {self.img_class: 0})
