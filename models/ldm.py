import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from datasets.dataset_utils import DataTypeEnum
from network import autoencoder_kl, clip, net_factory, uvit, uvit_t2i
from network.layers import layer_utils

from . import FlowMatching, model_factory, model_utils
from .base_model import BaseModel
from .ema import EMA


# @model_factory.register_model(name="ldm_flow_matching")
class BaseLDM:

    def __init__(self, config):
        self.init_autoencoder(config)
        if config.model.conditional:
            self.condition_type = config.model.condition.condition_type
            if self.condition_type == "text":
                self.init_pretrained_clip(config)

    def init_autoencoder(self, config):
        self.autoencoder = net_factory.create_network(config, "autoencoder_name").to(config.device)
        # self.network = net_factory.create_network(config).to(config.device)

    def init_pretrained_clip(self, config):
        self.clip = clip.FrozenCLIPEmbedder(config.model.clip.pretrained_path).to(config.device)

    def autoencoder_sample(self, moments):
        return self.autoencoder.sample(moments)

    @torch.cuda.amp.autocast()
    def autoencoder_decode(self, z):
        return self.autoencoder.decode(z)

    @torch.cuda.amp.autocast()
    def autoencoder_encode(self, x):
        return self.autoencoder.encode(x)

    def preprocess(self, x, y, data_type):
        # print(f"is_raw_data:{is_raw_data}")
        if data_type == DataTypeEnum.RAW_DATA:
            return self.autoencoder_encode(x), y
        return self.autoencoder_sample(x), y

    def unpreprocess(self, samples):
        return self.autoencoder_decode(samples)


@model_factory.register_model(name="fm_ldm")
class FM_LDM(BaseLDM, FlowMatching):
    def __init__(self, config):
        # print(self.__class__.mro())
        super(BaseLDM, self).__init__(config)
        super(FM_LDM, self).__init__(config)

    def init_parameters(self, config):
        super(BaseLDM, self).init_parameters(config)
        network_config = config.model[config.model.nn_name]
        self.img_size = (network_config.img_size, network_config.img_size)
        self.img_channels = network_config.in_chans

    def encode_text_condition(self, text_condition):
        return self.clip.encode(text_condition)
