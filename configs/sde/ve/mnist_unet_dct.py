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
    training = config.training
    training.model_checkpoint = "./data/checkpoints/generative_model/vesde/unet-mnist-14000-model"
    #training.optim_checkpoint = ".\data\checkpoints\ddpm-cifar10-81000-optim.pth"

    model = config.model
    model.channel_mults = (1, 2)
    model.name = "vesde"
    model.nn_name = "unet"
    model.sigma_max = 50
    model.sigma_min = 0.01
    model.num_scales = 232
    model.sigma_dist = "geometric"
    #model.norm = "gn"
    #model.activation = "elu"
    model.ngf = 128
    model.debug_groups = 10
    model.enable_debug = True
    model.predict_type = "noise"

    training = config.training
    training.anneal_power = 2
    training.log_all_sigmas = False
    training.batch_size = 48
    training.epochs = 100
    training.snapshot_freq = 1000
    training.log_freq = 10
    training.eval_freq = 10
    training.continuous = False

    optim = config.optim

    test = config.test
    test.save_path = "./data/test/"

    test.batch_size = 64
    test.num_samples = 1

    sampling = config.sampling
    sampling.step_lr = 0.0000062
    sampling.n_steps_each = 5
    sampling.final_only = True
    sampling.denoise = True
    sampling.sample_steps = 1000
    sampling.snr = 0.16
    sampling.method = 'pc'
    ##  euler  reversediffusion     ancestralsampling
    sampling.predictor = 'reversediffusion'
    ## ald  langevin
    sampling.corrector = 'ald'

    data = config.data
    data.logit_transform = False
    data.random_flip = True
    data.rescaled = True

    config.pipeline = "SDEPipeLine"

    return config
