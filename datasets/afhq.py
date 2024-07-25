import logging
from typing import Any, Callable, Dict, List, Tuple
import torch
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
from torchvision import datasets
from .data_trans import  resize_trans


class Afhq(datasets.ImageFolder):
    def __init__(self, root,  transform, img_class="all"):
        self.img_class = img_class
        super().__init__(root,transform= transform)


    def find_classes(self, directory):
        if self.img_class == "all":
            return super().find_classes(directory)
        return ([self.img_class], {self.img_class:0})
    @staticmethod
    def get_loader(root, batch_size, img_size, img_class="all"):

        if img_class == "all":
            train_dataset = Afhq(
                root= root + "/train/",
                transform=resize_trans(img_size),
                img_class= img_class
            )
            val_dataset = Afhq(
                root= root + "/val/",
                transform=resize_trans(img_size),
                img_class= img_class
            )
        else:
            train_dataset = Afhq(
                root=root + "/train/" ,
                transform=resize_trans(img_size),
                img_class= img_class
                # transform=script_utils.get_transform(),
            )

            val_dataset = Afhq(
                root=root + "/val/" ,
                transform=resize_trans(img_size),
                img_class= img_class
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
