import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin,
                                           register_to_config)
from diffusers.utils import BaseOutput, logging

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (DownBlock2D,
                          UNetMidStyleBlock2D,
                          UpBlock2D,
                          get_down_block,
                          get_up_block)


logger = logging.get_logger(__name__)


@dataclass
class UNetOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet(ModelMixin, ConfigMixin):
    """Conditional-diffusion U-Net of MSCI-Diffusion.

    Structure (paper Section 3.2 / 3.3):
      - The down path is a plain convolutional down path.
      - The mid block injects the multi-source style representation e_ms
        through cross-attention.
      - At every decoder resolution level, the HSCIM coordinates the content
        feature e_c^i from the content encoder with the denoising feature r_i
        propagated from the preceding block, and guides it with e_ms through
        cross-attention (coarse-to-fine style injection).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,  # 生成图像的分辨率
            in_channels: int = 4,  # 输入通道数
            out_channels: int = 4,  # 输出通道数
            flip_sin_to_cos: bool = True,  # 正弦位置编码选项
            freq_shift: int = 0,  # 频率偏移
            down_block_types: Tuple[str] = None,  # 下采样块类型
            up_block_types: Tuple[str] = None,  # 上采样块类型
            block_out_channels: Tuple[int] = (64, 128, 256, 512),  # 各层输出通道数
            layers_per_block: int = 1,  # 每块层数
            downsample_padding: int = 1,  # 下采样 padding
            mid_block_scale_factor: float = 1,  # 中间块缩放因子
            act_fn: str = "silu",  # 激活函数
            norm_num_groups: int = 32,  # GroupNorm 组数
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1024,  # 风格特征通道数 Cs
            attention_head_dim: int = 8,
            channel_attn: bool = False,  # 兼容旧参数（未使用）
            content_encoder_downsample_size: int = 4,  # 兼容旧参数（未使用）
            content_start_channel: int = 64,  # 内容编码器首层通道数
            content_resolution: int = 128,  # 内容图像分辨率（用于匹配内容金字塔）
            reduction: int = 32,  # HSCIM 通道注意力降维比例
    ):
        super().__init__()

        if down_block_types is None:
            down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D")
        if up_block_types is None:
            up_block_types = ("HSCIMUpBlock2D", "HSCIMUpBlock2D", "HSCIMUpBlock2D", "HSCIMUpBlock2D")

        self.sample_size = sample_size
        self.content_resolution = content_resolution
        self.content_start_channel = content_start_channel
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            print("Load the down block ", down_block_type)
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                content_channel=0,
                reduction=reduction,
                channel_attn=channel_attn,
            )
            self.down_blocks.append(down_block)

        # mid: style (e_ms) cross-attention at the bottleneck
        self.mid_block = UNetMidStyleBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
        )

        # count how many layers downsample the feature maps
        num_downsamplers = len(block_out_channels) - 1

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.up_content_channels = []  # 每个解码层级的期望内容特征通道数 Cc
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
            else:
                add_upsample = False

            # 该解码层级处理的分辨率：sample_size / 2^(num_downsamplers - i)
            level_resolution = sample_size // (2 ** (num_downsamplers - i)) if sample_size else None
            content_channel = self._content_channels_at(level_resolution)
            self.up_content_channels.append(content_channel)

            print("Load the up block ", up_block_type,
                  f"(level resolution {level_resolution}, content channels {content_channel})")
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,  # larger 1 than the down block
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                upblock_index=i,
                content_channel=content_channel,
                style_channels=cross_attention_dim,
                reduction=reduction,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def _content_channels_at(self, level_resolution: Optional[int]) -> int:
        """内容金字塔在分辨率 R 处的特征通道数：
        R = C        -> 3 (源内容图像，金字塔最浅层)
        R = C / 2^k  -> content_start_channel * 2^(k-1), k >= 1 (编码器特征)
        """
        C = self.content_resolution
        ch = self.content_start_channel
        if level_resolution is None or level_resolution >= C:
            return 3
        k = int(round(math.log2(C / max(level_resolution, 1))))
        k = max(1, min(k, 5))  # 编码器金字塔最多 5 级 (C/2 ... C/32)
        return ch * (2 ** (k - 1))

    @staticmethod
    def _pick_content_feature(pyramid: List[torch.Tensor],
                              spatial_size: Tuple[int, int],
                              expected_channels: int) -> torch.Tensor:
        """按分辨率从内容特征金字塔中选取当前解码层级的特征 e_c^i。"""
        # 1) 空间尺寸与通道数完全匹配
        for feature in pyramid:
            if feature.shape[-2:] == tuple(spatial_size) and feature.shape[1] == expected_channels:
                return feature
        # 2) 空间尺寸匹配
        for feature in pyramid:
            if feature.shape[-2:] == tuple(spatial_size):
                return feature
        # 3) 通道数匹配且面积最接近，随后双线性插值对齐
        candidates = [f for f in pyramid if f.shape[1] == expected_channels]
        if not candidates:
            candidates = list(pyramid)
        target_area = spatial_size[0] * spatial_size[1]
        feature = min(candidates,
                      key=lambda f: abs(f.shape[-2] * f.shape[-1] - target_area))
        return F.interpolate(feature, size=tuple(spatial_size), mode="bilinear", align_corners=False)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )
        if slice_size is not None and slice_size > self.config.attention_head_dim:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )

        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

        self.mid_block.set_attention_slice(slice_size)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DownBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states=None,
        content_encoder_downsample_size: int = 4,
        return_dict: bool = False,
    ) -> Union[UNetOutput, torch.FloatTensor]:
        """encoder_hidden_states: [e_ms (B, Cs, Hs, Ws), content pyramid (list of tensors)]"""
        style_feature = encoder_hidden_states[0]           # e_ms
        content_pyramid = encoder_hidden_states[1]         # {e_c^i}

        # e_ms 的 token 序列（供中间块交叉注意力作为 context）
        batch_size, style_channels, hs, ws = style_feature.shape
        style_tokens = style_feature.permute(0, 2, 3, 1).reshape(batch_size, hs * ws, style_channels)

        default_overall_up_factor = 2 ** (len(self.up_blocks) - 1)

        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples  # 保存特征用于跳跃连接

        # 4. mid（e_ms 交叉注意力）
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb, encoder_hidden_states=style_tokens)

        # 5. up（每个解码层级进行 HSCIM 风格-内容交互）
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            content_feature = self._pick_content_feature(
                content_pyramid, sample.shape[-2:], self.up_content_channels[i])

            if hasattr(upsample_block, "hscim"):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    content_feature=content_feature,
                    style_feature=style_feature,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return sample

        return UNetOutput(sample=sample)
