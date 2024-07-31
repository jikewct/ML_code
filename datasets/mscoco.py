import logging
import os

import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from . import dataset_factory
from .base_dataset import BaseDataset


@dataset_factory.register_dataset(name="mscoco")
class MSCOCO(BaseDataset):

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def init_dataset(self, **kargs):

        data_year = kargs["mscoco"].year
        self.train_dataset = datasets.coco.CocoCaptions(
            root=os.path.join(self.root_path + data_year, f"train{data_year}"),
            annFile=os.path.join(self.root_path + data_year, f"annotations/captions_train{data_year}.json"),
            transform=self.center_crop(self.img_size),
        )

        self.val_dataset = datasets.coco.CocoCaptions(
            root=os.path.join(self.root_path + data_year, f"val{data_year}"),
            annFile=os.path.join(self.root_path + data_year, f"annotations/captions_val{data_year}.json"),
            transform=self.center_crop(self.img_size),
        )
