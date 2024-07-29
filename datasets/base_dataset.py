import logging
import os

import numpy as np
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from .dataset_utils import DataTypeEnum


class BaseDataset(object):

    def __init__(self, root_path, img_size, img_channels, centered, **kargs):

        self.root_path = root_path
        self.img_size = img_size
        self.img_channels = img_channels
        self.centered = centered
        self.__dict__.update(kargs)
        self.init_trans()
        self.init_dataset(**kargs)

        # self.init_dataloader(**kargs)

    # def get_split(self, split, labeled=False):
    #     if split == "train":
    #         dataset = self.train
    #     elif split == "test":
    #         dataset = self.test
    #     else:
    #         raise ValueError

    #     if self.has_label:
    #         return dataset if labeled else UnlabeledDataset(dataset)
    #     else:
    #         assert not labeled
    #         return dataset

    def init_trans(self):
        imgCenter = tforms.Lambda(lambda x: 2 * x - 1)
        self.trans = tforms.Compose([tforms.ToTensor(), imgCenter])
        self.resize_trans = lambda img_size: tforms.Compose([tforms.Resize(img_size), tforms.ToTensor(), imgCenter])

    def init_dataset(self, **kargs):
        pass

    def get_dataloader(self, batch_size):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            drop_last=True,
        )

        return train_loader, val_loader

    def states(self):
        state = {"train_images": len(self.train_dataset), "val_images": len(self.val_dataset)}
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float, str, bool, bytes, bytearray, complex)):
                state[key] = val
        return state

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.0)
        v.clamp_(0.0, 1.0)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    @property
    def data_type(self):
        if hasattr(self.train_dataset, "data_type"):
            return self.train_dataset.data_type
        return DataTypeEnum.RAW_DATA

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


class BaseFeature(datasets.VisionDataset):
    def __init__(self, root_path, transform=None, target_transform=None):
        super().__init__(root_path, transform=transform, target_transform=target_transform)
        self.samples, self.labels = self.make_datasets()

    @property
    def data_type(self):
        return DataTypeEnum.FEATURE

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index], self.labels[index]

    def make_datasets(self):
        feature_path = os.path.join(self.root, "features.npy")
        label_path = os.path.join(self.root, "labels.npy")
        samples, labels = np.load(feature_path, allow_pickle=True), np.load(label_path, allow_pickle=True)
        return samples, labels


class BaseFeatureDataset(BaseDataset):

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def init_dataset(self, **kargs):
        self.train_dataset = BaseFeature(root_path=os.path.join(self.root_path, "train"))
        self.val_dataset = BaseFeature(root_path=os.path.join(self.root_path, "val"))
