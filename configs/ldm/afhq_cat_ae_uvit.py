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
    c(config, "training").update(
        batch_size=32,
        epochs=10000,
        snapshot_freq=500,
        log_freq=50,
        eval_freq=500,
        test_metric_freq=50000,
        resume=True,
        # model_checkpoint="./data/checkpoints/generative_model/flowMatching/uvit/afhq/96X96/30-network.pth",
    )
    c(config, "data").update(
        dataset="afhq_32x32_feature",
        img_size=(32, 32),
        img_channels=4,
        root_path="/home/jikewct/public/jikewct/Dataset/afhq/afhq256_features",
    )
    c(config, "data", "afhq_32x32_feature").update()
    c(config, "model").update(
        name="fm_ldm",
        nn_name="uvit",
        autoencoder_name="autoencoder_kl",
        grad_checkpoint=True,
    )
    c(config, "model", "fm_ldm").update()
    c(config, "model", "uvit").update(
        # cacl from autoencoder_kl ch_mult config
        img_size=config.data.img_size[0],
        patch_size=2,
        in_chans=config.data.img_channels,
        embed_dim=512,
        depth=16,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
        use_checkpoint=config.model.grad_checkpoint,
    )
    c(config, "model", "autoencoder_kl").update(
        double_z=True,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        embed_dim=4,
        pretrained_path="/home/jikewct/public/jikewct/Model/stable_diffusion/stable-diffusion/autoencoder_kl.pth",
        scale_factor=0.18215,
    )

    c(config, "sampling").update(
        log_freq=1,
        method="rk45",
        sampling_steps=50,
        denoise=True,
    )
    c(config, "sampling", "rk45").update(
        rtol=1e-3,
        atol=1e-3,
    )
    c(config, "sampling", "ode").update()

    c(config, "lr_scheduler").update(
        name="customized",
    )
    c(config, "lr_scheduler", "customized").update(
        warmup_steps=2000,
    )

    config.pipeline = "LDMPipeLine"

    return config
