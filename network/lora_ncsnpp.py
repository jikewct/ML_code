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

# pylint: skip-file

import functools
import logging

import numpy as np
import torch
import torch.nn as nn

from . import layers, layerspp, lora_layers, net_utils, normalization

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
LoraResnetBlockBigGAN = layerspp.LoRAResnetBlockBigGANpp

Combine = layerspp.Combine
conv3x3 = layers.ddpm_conv3x3
lora_conv3x3 = layers.lora_ddpm_conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
# get_normalization = normalization.get_normalization
default_initializer = net_utils.default_init


@net_utils.register_network(name="lora_ncsnpp")
class LoRANCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config.model.activation)
        self.register_buffer("sigmas", net_utils.get_sigmas(config))

        self.nf = nf = config.model.nf
        ch_mult = config.model.channel_mults
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attention_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.img_size[0] // (2**i) for i in range(num_resolutions)]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = config.model.progressive_input.lower()
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        self.lora_dim = config.model.lora_dim
        self.lora_alpha = config.model.lora_alpha
        self.lora_dropout = config.model.lora_dropout
        self.enable_lora = config.model.enable_lora
        self.grad_checkpoint = config.model.grad_checkpoint

        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            # assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale))
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(
                lora_layers.LoRAMergedLinear(
                    embed_dim, nf * 4, self.lora_dim, self.lora_alpha, self.lora_dropout, self.enable_lora, fan_in_fan_out=False, merge_weights=False
                )
            )
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(
                lora_layers.LoRAMergedLinear(
                    nf * 4, nf * 4, self.lora_dim, self.lora_alpha, self.lora_dropout, self.enable_lora, fan_in_fan_out=False, merge_weights=False
                )
            )
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        LoRAAttnBlock = functools.partial(
            layerspp.LoRAAttnBlockpp,
            init_scale=init_scale,
            skip_rescale=skip_rescale,
            r=self.lora_dim,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            merge_weights=False,
        )

        Upsample = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == "residual":
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM, act=act, dropout=dropout, init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")
        LoraResnetBlock = functools.partial(
            LoraResnetBlockBigGAN,
            act=act,
            dropout=dropout,
            fir=fir,
            fir_kernel=fir_kernel,
            init_scale=init_scale,
            skip_rescale=skip_rescale,
            temb_dim=nf * 4,
            r=self.lora_dim,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            grad_checkpoint=self.grad_checkpoint,
        )
        # Downsampling block

        channels = config.data.img_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(LoraResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(LoRAAttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(LoraResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(LoraResnetBlock(in_ch=in_ch))
        modules.append(LoRAAttnBlock(channels=in_ch))
        modules.append(LoraResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(LoraResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(LoRAAttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            lora_conv3x3(
                                in_ch,
                                channels,
                                bias=True,
                                init_scale=init_scale,
                                r=self.lora_dim,
                                lora_alpha=self.lora_alpha,
                                lora_dropout=self.lora_dropout,
                                merge_weights=False,
                            )
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(LoraResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, y=None):
        # timestep/noise_level embedding; only for continuous training
        # logging.info(f"x_max:{x.max()},x_min:{x.min()}, x_mean:{x.mean()}")
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            # logging.info(f"used_sigmas:{used_sigmas.mean()}")
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
            # logging.info(f"temb_norm:{torch.norm(temb)}, temb_shape:{temb.shape}")

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = net_utils.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        # logging.info(f"h_norm:{torch.norm(h)}, h_shape:{h.shape}")
        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            # logging.info(f"h_norm:{torch.norm(h)}, h_shape:{h.shape}")

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        # logging.info(f"pyramid_norm:{torch.norm(pyramid)}, pyramid_shape:{pyramid.shape}")

                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        # logging.info(f"pyramid_norm:{torch.norm(pyramid)}, pyramid_shape:{pyramid.shape}")

                        pyramid_h = self.act(modules[m_idx](h))
                        # logging.info(f"pyramid_norm:{torch.norm(pyramid)}, pyramid_shape:{pyramid.shape}")

                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        # logging.info(f"pyramid_norm:{torch.norm(pyramid)}, pyramid_shape:{pyramid.shape}")

                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                        # logging.info(f"pyramid_norm:{torch.norm(pyramid)}, pyramid_shape:{pyramid.shape}")

                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs
        # logging.info(f"h_norm:{torch.norm(h)}, h_shape:{h.shape}")

        if self.progressive == "output_skip":
            h = pyramid
            # logging.info(f"h_norm:{torch.norm(h)}, h_shape:{h.shape}")

        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas
            # logging.info(f"h_norm:{torch.norm(h[0])}")
        return h, {}
