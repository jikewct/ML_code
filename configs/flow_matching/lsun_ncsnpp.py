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

from configs.default_lsun_configs import get_default_configs


def get_config():
    config = get_default_configs()
    training = config.training
    model = config.model
    sampling = config.sampling
    data = config.data
    fast_fid = config.fast_fid
    optim = config.optim
    test = config.test

    #training.model_checkpoint = "E:\jikewct\Model\\rect_flow\cvt_model.pth"
    #training.model_checkpoint = "/home/jikewct/public/jikewct/Model/rect_flow/CelebA-HQ-Pytorch_model_ema.pth"
    training.continuous = False
    training.batch_size = 8
    training.epochs = 100
    training.snapshot_freq = 1000
    training.log_freq = 100
    training.eval_freq = 4000
    training.test_metric_freq = 10000

    #training.model_checkpoint = "./data/checkpoints/generative_model/flowMatching/ncsnpp-cifar10-14000-model"
    #training.optim_checkpoint = ".\data\checkpoints\ddpm-cifar10-81000-optim.pth"

    model.name = "flowMatching"
    model.nn_name = "ncsnpp"

    #model.norm = "gn"
    #model.activation = "elu"

    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.dropout = 0.
    model.norm = 'GroupNorm'
    model.activation = 'swish'
    model.nf = 128
    model.channel_mults = (1, 1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attention_resolutions = (12, )
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.embedding_type = 'fourier'
    model.fourier_scale = 16
    model.conv_size = 3

    model.sigma_dist = "geometric"
    model.sigma_max = 380
    model.sigma_min = 0.01
    model.num_scales = 2000

    data.category = "church_outdoor_train"
    data.img_size = (96, 96)

    test.save_path = "./data/test/"
    test.batch_size = 10
    test.num_samples = 2

    fast_fid.num_samples = 10000
    fast_fid.begin_step = 5000
    fast_fid.end_step = 1000000
    fast_fid.batch_size = 32

    sampling.sample_steps = 10
    sampling.denoise = True
    sampling.log_freq = 1
    sampling.rtol = 1e-3
    sampling.atol = 1e-3
    sampling.method = 'ode'

    config.pipeline = "FlowMatchingPipeLine"

    return config
