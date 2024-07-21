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

from configs.default_cifar10_configs import get_default_configs


def get_config():
    config = get_default_configs()
    training = config.training
    training = config.training
    model = config.model
    sampling = config.sampling
    data = config.data
    fast_fid = config.fast_fid
    optim = config.optim
    test = config.test
    training.model_checkpoint = "./data/checkpoints/generative_model/flowMatching/unet-cifar10-65000-model"
    #training.optim_checkpoint = ".\data\checkpoints\ddpm-cifar10-81000-optim.pth"

    model = config.model
    model.channel_mults = (1, 2, 2, 2)
    model.name = "flowMatching"
    model.nn_name = "unet"
    model.sigma_dist = "geometric"
    model.sigma_max = 50
    model.sigma_min = 0.01
    model.num_scales = 1000
    #model.norm = "gn"
    #model.activation = "elu"

    training.batch_size = 32
    training.epochs = 100
    training.snapshot_freq = 1000
    training.log_freq = 100
    training.eval_freq = 1000
    training.test_metric_freq = 50000

    test = config.test
    test.save_path = "./data/test/"
    test.batch_size = 64
    test.num_samples = 100

    sampling = config.sampling
    sampling.sample_steps = 10
    sampling.denoise = True
    sampling.log_freq = 1
    sampling.method = "ode"
    sampling.rtol = 1e-3
    sampling.atol = 1e-3
    config.pipeline = "FlowMatchingPipeLine"

    return config
