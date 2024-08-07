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
from configs.default_mnist_configs import get_default_configs


def get_config():
    config = get_default_configs()
    c(config, "training").update(
        batch_size=64,
        epochs=10000,
        snapshot_freq=500,
        log_freq=50,
        eval_freq=500,
        test_metric_freq=1000000,
        resume=True,
        continuous=False,
        # resume_path="/home/jikewct/public/jikewct/Repos/ml_code/data/checkpoints/generative_model/fm_ldm/uvit/afhq_32x32_feature/32X32",
        # model_checkpoint="/home/jikewct/public/jikewct/Model/uvit/mscoco_uvit_small.pth",
    )
    c(config, "data").update(
        num_classes=11,
    )
    c(config, "model").update(
        name="vesde",
        nn_name="uvit",
        conditional=True,
        grad_checkpoint=True,
    )
    c(config, "model", "condition").update(
        condition_type="class",
        cond_embedder_name="",
        cfg=True,
        p_cond=0.1,
        empty_latent_path="",
        empty_label=10,
    )
    c(config, "model", "vesde").update(
        scheduler="vens",  ## vpns,vens, rfns
    )
    c(config, "model", "vens").update(
        schedule_type="geo",  ## sd, linear
        num_scales=232,
        std_min=0.01,
        std_max=50,
    )

    c(config, "model", "uvit").update(
        # cacl from autoencoder_kl ch_mult config
        img_size=config.data.img_size[0],
        patch_size=2,
        in_chans=config.data.img_channels,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=config.data.num_classes,
        use_checkpoint=config.model.grad_checkpoint,
    )
    c(config, "test").update(
        batch_size=5,
        num_samples=5,
        save_path="./data/test/",
    )
    c(config, "sampling").update(
        log_freq=1,
        method="pc",  # ode, numerical, pc, dpm_solver
        denoise=True,
        sampling_conditions=[6, 8, 9],
        guidance_scale=1.0,
    )
    c(config, "sampling", "numerical").update(
        rtol=1e-3,
        atol=1e-3,
        equation_type="ode",
        # ode in (eluer, rk45),
        # sde in 'euler', 'euler_heun', 'heun', 'log_ode', 'midpoint', 'milstein', 'reversible_heun', 'srk']
        method="rk45",
        sampling_steps=20,
    )
    c(config, "sampling", "ode").update(
        sampling_steps=100,
    )
    c(config, "sampling", "dpm_solver").update(
        sampling_steps=50,
    )
    c(config, "sampling", "pc").update(
        predictor="reversediffusion",  # euler,reversediffusion,ancestralsampling,""
        corrector="ald",  # ald,langevin, ""
        n_step_each=1,
        snr=0.16,
    )
    c(config, "optim").update(
        optimizer="adamw",
    )
    c(config, "optim", "adamw").update(
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )
    c(config, "lr_scheduler").update(
        name="customized",
    )
    c(config, "lr_scheduler", "customized").update(
        warmup_steps=2000,
    )

    config.pipeline = "SDEPipeLine"

    return config
