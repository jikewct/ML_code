import ml_collections
import ml_collections.config_dict
import torch

from configs.config_utils import *


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    c(config, "training")
    # sampling
    c(config, "sampling")
    # evaluation
    c(config, "eval")
    c(config, "test")
    c(config, "fast_fid").update(
        ds_state_file="/home/jikewct/public/jikewct/Dataset/afhq/train/train_cat_fid_stats.npz",
    )
    # data
    c(config, "data").update(
        dataset="afhq",
        img_size=(32, 32),
        img_channels=3,
        num_classes=3,
        # img_class="",
        root_path="/home/jikewct/public/jikewct/Dataset/afhq",
        random_flip=True,
        centered=True,
        uniform_dequantization=False,
    )
    c(config, "data", "afhq").update()

    # model
    c(config, "model")
    c(config, "model", "ema")

    # optimization
    c(config, "optim").update(
        optimizer="adamw",
    )
    c(config, "optim", "adamw")
    c(config, "lr_scheduler").update(
        name="customized",
    )
    c(config, "lr_scheduler", "customized")

    c(config, "seed")
    c(config, "device")
    c(config, "pipeline")
    return config
