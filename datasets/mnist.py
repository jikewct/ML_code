import logging

import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from . import dataset_factory
from .base_dataset import BaseDataset


@dataset_factory.register_dataset(name="mnist")
class Mnist(BaseDataset):

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def init_dataset(self, **kargs):

        self.train_dataset = datasets.MNIST(
            root=self.root_path,
            train=True,
            download=True,
            transform=self.trans,
        )

        self.val_dataset = datasets.MNIST(
            root=self.root_path,
            train=False,
            download=True,
            transform=self.trans,
        )
