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
    training.model_checkpoint = "./data/checkpoints/generative_model/smld/unet-cifar10-43000-model"
    model = config.model
    model.channel_mults = (1, 2, 2, 2)
    model.name = "smld"
    model.nn_name = "unet"
    model.sigma_max = 50
    model.sigma_min = 0.01
    model.num_scales = 232
    model.sigma_dist = "geometric"
    #model.norm = "gn"
    #model.activation = "elu"
    model.ngf = 128
    #score or noise or union
    model.predict_type = "noise"
    model.union_tau = 1
    model.union_threshold = 150

    training = config.training
    training.anneal_power = 2
    training.log_all_sigmas = False
    training.batch_size = 32
    training.epochs = 100

    training.snapshot_freq = 1000
    training.log_freq = 100
    training.eval_freq = 1000

    optim = config.optim
    optim.lr = 2e-4
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.amsgrad = False
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    test = config.test
    test.save_path = "./data/test/"

    test.batch_size = 64
    test.num_samples = 1
    test.sample_steps = 1000

    sampling = config.sampling
    sampling.step_lr = 0.0000062
    sampling.n_steps_each = 5
    sampling.final_only = True
    sampling.denoise = True

    data = config.data
    data.logit_transform = False
    data.random_flip = True
    data.rescaled = True

    config.pipeline = "SMLDPipeLine"

    return config
