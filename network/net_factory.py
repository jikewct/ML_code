# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""All functions and modules related to model definition.
"""

import logging
import math
import string
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NETS = {}


def register_network(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _NETS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _NETS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_network(name):
    return _NETS[name]


def create_network(config, key="nn_name") -> nn.Module:
    """Create the score model."""
    network_name = config.model[key]
    if hasattr(config.model, network_name):
        net = get_network(network_name)(**config.model[network_name])
    else:
        net = get_network(network_name)(config)
    net = net.to(config.device)
    num_params = 0
    for p in net.parameters():
        num_params += p.numel()
    logging.info(f"network:{network_name},Number of Parameters:{num_params}")

    # score_model = torch.nn.DataParallel(score_model)
    return net
