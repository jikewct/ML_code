import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from datasets import DataTypeEnum
from network import *

from . import model_factory
from .flow_matching import FlowMatching
from .sde import VPSDE


# @model_factory.register_model(name="ldm_flow_matching")
class LDM:

    def __init__(self, config):
        self.init_autoencoder(config)

    def init_autoencoder(self, config):
        class_name = LDM.__name__.lower()
        self.autoencoder = net_factory.create_network(config.model[class_name], "autoencoder_name").to(config.device)
        # self.network = net_factory.create_network(config).to(config.device)

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
class FM_LDM(LDM, FlowMatching):
    def __init__(self, config):
        # print(self.__class__.mro())
        super(LDM, self).__init__(config)
        super(FM_LDM, self).__init__(config)

    def init_parameters(self, config):
        super(LDM, self).init_parameters(config)
        network_config = config.model[config.model.nn_name]
        self.img_size = (network_config.img_size, network_config.img_size)
        self.img_channels = network_config.in_chans


@model_factory.register_model(name="vpsde_ldm")
class VPSDE_LDM(LDM, VPSDE):
    def __init__(self, config):
        # print(self.__class__.mro())
        super(LDM, self).__init__(config)
        super(VPSDE_LDM, self).__init__(config)

    def init_parameters(self, config):
        super(LDM, self).init_parameters(config)
        network_config = config.model[config.model.nn_name]
        self.img_size = (network_config.img_size, network_config.img_size)
        self.img_channels = network_config.in_chans
