import logging

import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from . import dataset_factory
from .base_dataset import BaseDataset
from .data_trans import resize_trans


@dataset_factory.register_dataset(name="lsun")
class Lsun(BaseDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def init_dataset(self, **kargs):

        self.train_dataset = datasets.LSUN(
            root=self.root_path,
            classes=["church_outdoor_train"],
            transform=resize_trans(self.img_size),
        )

        self.val_dataset = datasets.MNIST(
            root=self.root_path,
            classes=["church_outdoor_val"],
            transform=resize_trans(self.img_size),
        )
