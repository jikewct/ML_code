import glob
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from . import dataset_factory
from .base_dataset import BaseFeature, BaseFeatureDataset
from .data_trans import resize_trans


@dataset_factory.register_dataset(name="mscoco_32x32_feature")
class MsCOCO_32X32_Feature(BaseFeatureDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def init_dataset(self, **kargs):
        self.train_dataset = MSCOCO_32X32_Feature(root_path=os.path.join(self.root_path, "train"))
        self.val_dataset = MSCOCO_32X32_Feature(root_path=os.path.join(self.root_path, "val"))


class MSCOCO_32X32_Feature(BaseFeature):
    def __init__(self, root_path):
        super().__init__(root_path)

    def __getitem__(self, index: int):
        data = np.load(self.samples[index])
        img, text = data["img"], data["text"]
        rand_index = np.random.randint(0, len(text))
        return img, text[rand_index]

    def make_datasets(self):
        files = glob.glob(os.path.join(self.root, "*.npz"))
        return files, []
