import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from . import dataset_factory
from .base_dataset import BaseFeatureDataset
from .data_trans import resize_trans


@dataset_factory.register_dataset(name="afhq_32x32_feature")
class Afhq_32X32_Feature(BaseFeatureDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    # def init_dataset(self, **kargs):
    #     self.train_dataset = AFHQ_32X32_Feature(root_path=os.path.join(self.root_path, "train"))
    #     self.val_dataset = AFHQ_32X32_Feature(root_path=os.path.join(self.root_path, "val"))


# class AFHQ_32X32_Feature(BaseFeature):
#     def __init__(self, root_path):
#         super().__init__(root_path)

#     def make_datasets(self):
#         feature_path = os.path.join(self.root, "features.npy")
#         label_path = os.path.join(self.root, "labels.npy")
#         samples, labels = np.load(feature_path, allow_pickle=True), np.load(label_path, allow_pickle=True)
#         return samples, labels
