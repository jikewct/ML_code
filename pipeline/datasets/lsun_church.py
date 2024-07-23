import logging
import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets

from .data_trans import  resize_trans


class Lsun:

    @staticmethod
    def get_loader(root, batch_size, img_size):

        train_dataset = datasets.LSUN(
            root=root,
            classes=["church_outdoor_train"],
            transform=resize_trans(img_size),
            # transform=script_utils.get_transform(),
        )

        val_dataset = datasets.LSUN(
            root=root,
            classes=["church_outdoor_val"],
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
        test_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=True,
        )

        return train_loader, test_loader
