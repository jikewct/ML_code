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

from configs.config_utils import *
from configs.default_afhq_configs import get_default_configs


def get_config():
    config = get_default_configs()
    training = config.training
    model = config.model
    sampling = config.sampling
    data = config.data
    fast_fid = config.fast_fid
    optim = config.optim
    test = config.test

    c(config, "training").update(
        model_checkpoint="./data/checkpoints/generative_model/flowMatching/ncsnpp-afhq-6000-model",
        # training.model_checkpoint = "/home/jikewct/public/jikewct/Model/rect_flow/CelebA-HQ-Pytorch_model_ema.pth",
        continuous=False,
        batch_size=16,
        epochs=100,
        snapshot_freq=1000,
        log_freq=20,
        eval_freq=500,
        test_metric_freq=10000,
    )
    c(config, "model").update(
        name="flowMatching",
        nn_name="ncsnpp",
        # model.norm = "gn"
        # model.activation = "elu"
    )
    c(config, "flowMatching").update(
        num_scales=1000,
    )
    # ncsnpp config
    c(config, "model").update(
        scale_by_sigma=False,
        ema_rate=0.999,
        dropout=0.0,
        norm="GroupNorm",
        activation="swish",
        nf=128,
        channel_mults=(1, 1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="output_skip",
        progressive_input="input_skip",
        progressive_combine="sum",
        attention_type="ddpm",
        init_scale=0.0,
        embedding_type="fourier",
        fourier_scale=16,
        conv_size=3,
        grad_checkpoint=True,
        sigma_dist="geometric",
        sigma_max=50,
        sigma_min=0.01,
        num_scales=1000,
    )
    # v(config, "model").update(v(config, "ncsnpp"))
    c(config, "data").update(
        img_size=(128, 128),
        img_class="cat",
    )

    c(config, "test").update(
        save_path="./data/test/",
        batch_size=10,
        num_samples=2,
    )
    c(config, "fast_fid").update(
        num_samples=10000,
        begin_step=5000,
        end_step=1000000,
        batch_size=32,
    )
    c(config, "sampling").update(
        log_freq=1,
        method="rk45",
        sampling_steps=50,
    )
    c(config.sampling, "rk45").update(
        rtol=1e-3,
        atol=1e-3,
    )
    c(config.sampling, "ode").update()
    c(config, "optim").update(
        lr=1e-4,
    )

    config.pipeline = "FlowMatchingPipeLine"

    return config
