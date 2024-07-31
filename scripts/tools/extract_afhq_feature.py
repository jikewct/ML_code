import logging
import os
import sys
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from absl import app

sys.path.append(".")
sys.path.append("..")
os.chdir("/home/jikewct/public/jikewct/Repos/ml_code")
from configs.config_utils import c
from configs.ldm.afhq_cat_ae_uvit import get_config
from datasets import dataset_factory, dataset_utils
from models import FlowMatching, model_factory, model_utils
from network import autoencoder_kl, net_factory
from network.layers import layer_utils


def extract_feature(argv):
    config = get_config()
    data_config = config.data
    config.data.update(
        dataset="afhq",
        img_size=(256, 256),
        img_channels=3,
        root_path="/home/jikewct/public/jikewct/Dataset/afhq",
    )
    c(config.data, "afhq").update(img_class="cat")
    print(config.model.nn_name, config.data)
    autoencoder = net_factory.create_network(config, "autoencoder_name").to(config.device)
    dataset = dataset_factory.create_dataset(config)
    print(dataset.states())
    train_loader, test_loader = dataset.get_dataloader(16)
    features = []
    labels = []
    for x, y in train_loader:
        x = x.to(config.device)
        moments = autoencoder(x, fn="encode_moments").detach().cpu().numpy()
        features.append(moments)
        labels.append(y)
        print(f"processed {len(features)} batches.....")
        # if len(features) > 2:
        #     break

    features_numpy = np.concatenate(features, axis=0)
    labels_numpy = np.concatenate(labels, axis=0)
    path = os.path.join(config.data.root_path, f"{data_config.dataset}{config.data.img_size[0]}_features/train/")
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "features.npy"), features_numpy)
    np.save(os.path.join(path, "labels.npy"), labels_numpy)
    print(features_numpy.shape, labels_numpy.shape)
    features = []
    labels = []
    os.makedirs(path, exist_ok=True)
    for x, y in test_loader:
        x = x.to(config.device)
        moments = autoencoder(x, fn="encode_moments").detach().cpu().numpy()
        features.append(moments)
        labels.append(y)
        print(f"processed {len(features)} batches.....")
        # if len(features) > 2:
        #     break

    features_numpy = np.concatenate(features, axis=0)
    labels_numpy = np.concatenate(labels, axis=0)
    path = os.path.join(config.data.root_path, f"{data_config.dataset}{config.data.img_size[0]}_features/val/")
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "features.npy"), features_numpy)
    np.save(os.path.join(path, "labels.npy"), labels_numpy)
    print(features_numpy.shape, labels_numpy.shape)


if __name__ == "__main__":
    app.run(extract_feature)
