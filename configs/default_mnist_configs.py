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
        ds_state_file="",
    )
    # data

    c(config, "data").update(
        dataset="mnist",
        img_size=(28, 28),
        img_channels=1,
        num_classes=10,
        # img_class="",
        root_path="/home/jikewct/public/jikewct/Dataset/mnist",
        random_flip=True,
        centered=True,
        uniform_dequantization=False,
    )
    c(config, "data", "mnist").update()
    # model
    c(config, "model")
    c(config, "model", "ema")
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
    # config.seed = 42
    # config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # config.pipeline = ""
    return config
