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

# Lint as: python3
"""Train the original DDPM model."""

from configs.default_mnist_configs import get_default_configs


def get_config():
    config = get_default_configs()
    config.pipeline = "DDPMPipeLine"
    training = config.training
    model = config.model
    sampling = config.sampling
    data = config.data
    fast_fid = config.fast_fid
    optim = config.optim
    test = config.test

    model.schedule = "linear"
    model.num_scales = 1000
    model.beta_min = 1e-4
    model.beta_max = 0.02

    return config
