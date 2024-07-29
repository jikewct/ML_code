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

from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()
    config.pipeline = "SMLDPipeLine"
    training = config.training
    model = config.model
    sampling = config.sampling
    data = config.data
    fast_fid = config.fast_fid
    optim = config.optim
    test = config.test

    ### 调整默认配置
    model.name = "smld"
    model.nn_name = "unet"
    sampling.step_lr = 0.0000062
    training.model_checkpoint = "./data/checkpoints/generative_model/smld/unet-mnist-40000-model"
    training.optim_checkpoint = ""
    training.batch_size = 48

    # model

    ### model 特有配置
    training.anneal_power = 2

    model.predict_type = "noise"
    model.union_tau = 1
    model.union_threshold = 150
    model.num_scales = 232
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.sigma_dist = "geometric"

    sampling.n_steps_each = 5
    sampling.final_only = True
    sampling.denoise = True

    return config
