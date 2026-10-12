import torch
from torch import nn
from torchvision.ops import DeformConv2d

from .attention import (SpatialTransformer,
                        OffsetRefStrucInter,
                        ChannelAttnBlock,
                        CAFM,
                        PPA)
from .hscim import HSCIM
from .resnet import (Downsample2D,
                     ResnetBlock2D,
                     Upsample2D)


def get_down_block(
        down_block_type,  # 指定要使用的下采样块类型，例如 "DownBlock2D" 或 "MCADownBlock2D"
        num_layers,  # 该块中包含的层数
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        temb_channels,  # 时间嵌入通道数（通常用于扩散模型）
        add_downsample,  # 是否添加下采样层
        resnet_eps,  # ResNet 层的 epsilon 参数（数值稳定性）
        resnet_act_fn,  # ResNet 层的激活函数
        attn_num_head_channels,  # 注意力机制中的头部通道数（用于多头注意力）
        resnet_groups=None,  # ResNet 组归一化的组数
        cross_attention_dim=None,  # 交叉注意力的输入维度（仅适用于 MCA 模块）
        downsample_padding=None,  # 下采样时的填充大小
        channel_attn=False,  # 是否启用通道注意力机制
        content_channel=32,  # 内容特征通道数（适用于 MCA 模块）
        reduction=32  # 通道注意力的降维因子（适用于 MCA 模块）
):
    # 如果 down_block_type 以 "UNetRes" 开头，则去掉前缀 "UNetRes"
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding)
    elif down_block_type == "MCADownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return MCADownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            channel_attn=channel_attn,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel,
            reduction=reduction)
    else:
        raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
        up_block_type,  # 指定上采样块的类型，例如 "UpBlock2D" 或 "MCAUpBlock2D"
        num_layers,  # 该上采样块包含的层数
        in_channels,  # 当前块的输入通道数
        out_channels,  # 当前块的输出通道数
        prev_output_channel,  # 上一层输出的通道数
        temb_channels,  # 时间嵌入通道数（用于扩散模型）
        add_upsample,  # 是否添加上采样操作（通常是 2x 上采样）
        resnet_eps,  # ResNet 层中的 epsilon 值（用于数值稳定性）
        resnet_act_fn,  # ResNet 层的激活函数（如 ReLU 或 SiLU）
        attn_num_head_channels,  # 多头注意力的头部通道数
        upblock_index,  # 该上采样块在整个 UNet 结构中的索引（控制不同阶段的行为）
        resnet_groups=None,  # ResNet 归一化层的组数
        cross_attention_dim=None,  # 交叉注意力机制的输入维度（适用于 MCA 结构）
        structure_feature_begin=64,  # 结构特征的起始通道数（可能与字体生成任务相关）
        content_channel=None,  # 当前层级内容特征通道数 Cc（HSCIM 模块使用）
        style_channels=None,  # 多源风格特征通道数 Cs（HSCIM 模块使用）
        reduction=32  # 通道注意力降维比例（HSCIM 模块使用）
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups)
    elif up_block_type == "HSCIMUpBlock2D":
        if cross_attention_dim is None or content_channel is None:
            raise ValueError("cross_attention_dim and content_channel must be specified for HSCIMUpBlock2D")
        return HSCIMUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel,
            style_channels=style_channels,
            reduction=reduction,
            upblock_index=upblock_index)
    elif up_block_type == "StyleRSIUpBlock2D":
        return StyleRSIUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            structure_feature_begin=structure_feature_begin,
            upblock_index=upblock_index)
    else:
        raise ValueError(f"{up_block_type} does not exist.")


class UNetMidMCABlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入通道数（特征图通道）
            temb_channels: int,  # 时间嵌入通道数（用于扩散模型）
            channel_attn: bool = False,  # 是否启用通道注意力
            dropout: float = 0.0,  # Dropout 概率
            num_layers: int = 1,  # 该模块中包含的层数
            resnet_eps: float = 1e-6,  # ResNet 归一化中的 epsilon 值
            resnet_time_scale_shift: str = "default",  # ResNet 的时间嵌入模式
            resnet_act_fn: str = "swish",  # ResNet 的激活函数（默认 SiLU/Swish）
            resnet_groups: int = 32,  # ResNet 的 GroupNorm 组数
            resnet_pre_norm: bool = True,  # 是否在 ResNet 之前进行归一化
            attn_num_head_channels=1,  # 注意力头部通道数
            attention_type="default",  # 注意力类型（未使用）
            output_scale_factor=1.0,  # 输出缩放因子
            cross_attention_dim=1280,  # 交叉注意力的输入维度
            content_channel=256,  # 内容特征的额外通道数
            reduction=32,  # 通道注意力的缩放因子
            **kwargs,  # 额外参数
    ):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        content_attentions = []
        style_attentions = []

        for _ in range(num_layers):
            content_attentions.append(
                # CAFM(
                #     in_channels=in_channels,  # 额外增加 content_channel 维度
                #     content_channel=content_channel,
                #     out_channels=in_channels,
                #     groups=resnet_groups,
                #     # non_linearity=resnet_act_fn,
                #     # channel_attn=channel_attn,
                #     reduction=reduction,
                #     #i=i
                # )
                ChannelAttnBlock(
                    in_channels=in_channels + content_channel,
                    out_channels=in_channels,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            style_attentions.append(
                SpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )

            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.content_attentions = nn.ModuleList(content_attentions)
        self.style_attentions = nn.ModuleList(style_attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
            self,
            hidden_states,  # 当前特征图 形状：(batch_size, in_channels, height, width)
            temb=None,
            encoder_hidden_states=None,  # 编码器特征（[style, content]）
            index=None,
    ):
        #通过 ResNet 处理初始特征。
        hidden_states = self.resnets[0](hidden_states, temb)
        for content_attn, style_attn, resnet in zip(self.content_attentions, self.style_attentions, self.resnets[1:]):
            # content
            current_content_feature = encoder_hidden_states[1][index]  # 获取当前层的内容特征。
            hidden_states = content_attn(hidden_states, current_content_feature)

            # t_embed
            hidden_states = resnet(hidden_states, temb)

            # style
            current_style_feature = encoder_hidden_states[0]  # 形状：(batch_size, channel, height, width)
            batch_size, channel, height, width = current_style_feature.shape
            current_style_feature = current_style_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width,
                                                                                      channel)
            hidden_states = style_attn(hidden_states, context=current_style_feature)

        return hidden_states


class MCADownBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            temb_channels: int,  # 时间嵌入（Timestep Embedding）的通道数
            dropout: float = 0.0,  # Dropout 比例，默认 0.0
            channel_attn: bool = False,  # 是否使用通道注意力（Channel Attention）
            num_layers: int = 1,  # ResNet 层数
            resnet_eps: float = 1e-6,  # ResNet 计算中的 epsilon 值（用于稳定计算）
            resnet_time_scale_shift: str = "default",  # ResNet 时间嵌入方式
            resnet_act_fn: str = "swish",  # ResNet 的激活函数
            resnet_groups: int = 32,  # ResNet 归一化的分组数
            resnet_pre_norm: bool = True,  # ResNet 是否在激活前进行归一化
            attn_num_head_channels=1,  # 计算风格注意力（Style Attention）时的头部通道数
            cross_attention_dim=1280,  # 风格注意力（Style Attention）的输入特征维度
            attention_type="default",  # 注意力类型
            output_scale_factor=1.0,  # 输出缩放因子
            downsample_padding=1,  # 下采样时的 padding 大小
            add_downsample=True,  # 是否在最后添加下采样层
            content_channel=16,  # 通道注意力的内容特征通道数
            reduction=32,  # 通道注意力的降维因子
    ):
        super().__init__()
        content_attentions = []
        resnets = []
        style_attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            content_attentions.append(
                # CAFM(
                #     in_channels=in_channels,  # 额外增加 content_channel 维度
                #     content_channel=content_channel,
                #     out_channels=in_channels,
                #     groups=resnet_groups,
                #     #non_linearity=resnet_act_fn,
                #     #channel_attn=channel_attn,
                #     reduction=reduction,
                #     i=i
                # )
                # PPA(
                #     in_features1=in_channels,
                #     in_features2=content_channel,
                #     filters=in_channels,
                # )
                ChannelAttnBlock(
                    in_channels=in_channels + content_channel,
                    out_channels=in_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            print("The style_attention cross attention dim in Down Block {} layer is {}".format(i + 1,
                                                                                                cross_attention_dim))
            style_attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
                # EUCB(
                #     in_channels=out_channels,
                #     out_channels=out_channels,
                # )
            )
        self.content_attentions = nn.ModuleList(content_attentions)
        self.style_attentions = nn.ModuleList(style_attentions)
        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,  #输入特征
            index,
            temb=None,
            encoder_hidden_states=None  #编码器的隐藏状态（包含风格和内容信息）
    ):
        output_states = ()

        for content_attn, resnet, style_attn in zip(self.content_attentions, self.resnets, self.style_attentions):
            # content
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)

            # t_embed
            hidden_states = resnet(hidden_states, temb)

            # style
            current_style_feature = encoder_hidden_states[0]
            batch_size, channel, height, width = current_style_feature.shape
            current_style_feature = current_style_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width,
                                                                                      channel)
            hidden_states = style_attn(hidden_states, context=current_style_feature)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_downsample=True,
            downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class StyleRSIUpBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 当前块输入的通道数
            out_channels: int,  # 当前块输出的通道数
            prev_output_channel: int,  # 前一层输出的通道数，用于第一层 ResNet
            temb_channels: int,  # 时间嵌入的通道数
            dropout: float = 0.0,  # dropout 率
            num_layers: int = 1,  # 该模块中的 ResNet 层数
            resnet_eps: float = 1e-6,  # ResNet 归一化的 epsilon 值
            resnet_time_scale_shift: str = "default",  # 时间步归一化策略
            resnet_act_fn: str = "swish",  # ResNet 的激活函数
            resnet_groups: int = 32,  # ResNet 归一化的分组数
            resnet_pre_norm: bool = True,  # ResNet 是否在计算前进行归一化
            attn_num_head_channels=1,  # 注意力头数
            cross_attention_dim=1280,  # 交叉注意力的维度
            attention_type="default",  # 注意力类型
            output_scale_factor=1.0,  # 输出缩放因子
            downsample_padding=1,  # 下采样 padding 值
            structure_feature_begin=64,  # 结构特征的起始通道数
            upblock_index=1,  # 该上采样块的索引（用于选取风格结构特征）
            add_upsample=True,  # 是否添加上采样层
            resolution=128
    ):
        super().__init__()
        resnets = []
        attentions = []
        sc_interpreter_offsets = []  # 偏移结构解释器
        dcn_deforms = []  # 变形卷积

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.upblock_index = upblock_index
        self.resolution = resolution

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 计算 ResNet 的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 偏移结构解释器（用于计算偏移量）
            sc_interpreter_offsets.append(
                OffsetRefStrucInter(
                    res_in_channels=res_skip_channels,
                    style_feat_in_channels=int(structure_feature_begin * 2 / upblock_index),
                    n_heads=attn_num_head_channels,
                    num_groups=resnet_groups,
                )
            )
            # 可变形卷积（用于自适应调整风格偏移）
            dcn_deforms.append(
                DeformConv2d(
                    in_channels=res_skip_channels,
                    out_channels=res_skip_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    dilation=1,
                )
            )

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,  # 每个头的通道数
                    depth=1,
                    context_dim=cross_attention_dim,  # 交叉注意力上下文维度
                    num_groups=resnet_groups,
                )
            )
        self.sc_interpreter_offsets = nn.ModuleList(sc_interpreter_offsets)
        self.dcn_deforms = nn.ModuleList(dcn_deforms)
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.num_layers = num_layers

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False  # 是否开启梯度检查点（用于节省显存）

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,  # 当前特征图
            res_hidden_states_tuple,  # 跳跃连接特征
            style_structure_features,  # 风格结构特征
            temb=None,  # 时间步嵌入
            encoder_hidden_states=None,  # 额外的编码器特征
            upsample_size=None,  # 目标上采样大小
    ):
        total_offset = 0

        style_content_feat = style_structure_features[-self.upblock_index - 2]
        # if self.resolution == 96:
        #     index_offset = -self.upblock_index - 2  # 比如 -3
        # elif self.resolution == 128:
        #     index_offset = -self.upblock_index - 3  # 比如 -2 或其他
        # style_content_feat = style_structure_features[index_offset]

        for i, (sc_inter_offset, dcn_deform, resnet, attn) in \
                enumerate(zip(self.sc_interpreter_offsets, self.dcn_deforms, self.resnets, self.attentions)):
            # 取出最近的跳跃连接特征。
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # 计算偏移量，调整跳跃连接特征。
            offset = sc_inter_offset(res_hidden_states, style_content_feat)
            offset = offset.contiguous()
            # offset sum
            offset_sum = torch.mean(torch.abs(offset))
            total_offset += offset_sum

            res_hidden_states = res_hidden_states.contiguous()
            res_hidden_states = dcn_deform(res_hidden_states, offset)
            # concat as input
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        offset_out = total_offset / self.num_layers

        return hidden_states, offset_out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
class UpBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UNetMidStyleBlock2D(nn.Module):
    """Mid block of the MSCI-Diffusion U-Net: ResNet - style cross-attention
    (with the multi-source style representation e_ms as context) - ResNet."""

    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attn_num_head_channels=1,
            output_scale_factor=1.0,
            cross_attention_dim=1024,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ),
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ),
        ])
        self.style_attention = SpatialTransformer(
            in_channels,
            attn_num_head_channels,
            in_channels // attn_num_head_channels,
            depth=1,
            context_dim=cross_attention_dim,
            num_groups=resnet_groups,
        )

    def set_attention_slice(self, slice_size):
        self.style_attention._set_attention_slice(slice_size)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        # encoder_hidden_states: style token sequence of e_ms, (B, Ns, Cs)
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.style_attention(hidden_states, context=encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class HSCIMUpBlock2D(nn.Module):
    """Up block of the MSCI-Diffusion U-Net. At the i-th resolution level of
    the decoder, the incoming denoising feature r_i interacts with the
    content feature e_c^i through the HSCIM (channel attention -> spatial
    attention -> concat fusion -> style-guided cross-attention with e_ms);
    the resulting feature is returned to the U-Net block for the subsequent
    denoising computations (paper Section 3.2 / 3.3)."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attn_num_head_channels=1,
            cross_attention_dim=1024,
            output_scale_factor=1.0,
            content_channel=32,
            style_channels=1024,
            reduction=32,
            upblock_index=0,
            add_upsample=True,
    ):
        super().__init__()
        self.upblock_index = upblock_index
        self.num_layers = num_layers

        # HSCIM at this resolution level; Cr is the channel width of the
        # denoising feature r_i entering the block.
        self.hscim = HSCIM(
            unet_channels=prev_output_channel,
            content_channels=content_channel,
            style_channels=style_channels,
            reduction=reduction,
        )

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,  # r_i: 从上一个 U-Net 块传递来的去噪特征
            res_hidden_states_tuple,  # 跳跃连接特征
            content_feature,  # e_c^i: 当前层级的内容特征
            style_feature,  # e_ms: 多源风格特征 (B, Cs, Hs, Ws)
            temb=None,
            upsample_size=None,
    ):
        # HSCIM: 层级化风格-内容交互（通道注意力 -> 空间注意力 -> 融合 -> 风格交叉注意力）
        interaction = self.hscim(hidden_states, content_feature, style_feature)
        hidden_states = hidden_states + interaction

        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
