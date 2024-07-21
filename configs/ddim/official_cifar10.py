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
    training.model_checkpoint = "E:\jikewct\Model\google\ddpm_ema_cifar10\model-790000.ckpt"
    training.optim_checkpoint = ""

    model = config.model
    model.channel_mults = (1, 2, 2, 2)
    model.name = "ddim"
    model.nn_name = "official_ddpm_unet"
    # diffrent from our model
    model.attention_resolutions = (16, )

    optim = config.optim
    optim.lr_schedule = ""

    test = config.test
    test.save_path = "./data/test/official_ddim"
    test.batch_size = 256
    test.num_samples = 45000
    test.sample_steps = 100
    test.eta = 0.0

    config.pipeline = "DDIMPipeLine"
    return config
