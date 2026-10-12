from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
            self,
            in_channels: int,
            n_heads: int,
            d_head: int,
            depth: int = 1,
            dropout: float = 0.0,
            num_groups: int = 32,
            context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)
            ]
        )

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight,
                                                                  inner_dim)  # here change the shape torch.Size([1, 4096, 128])
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)  # hidden_states: torch.Size([1, 4096, 128])
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1,
                                                                                        2)  # torch.Size([1, 128, 64, 64])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
            self,
            dim: int,
            n_heads: int,
            d_head: int,
            dropout=0.0,
            context_dim: Optional[int] = None,
            gated_ff: bool = True,
            checkpoint: bool = True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        hidden_states = self.attn1(
            self.norm1(hidden_states)) + hidden_states  # hidden_states: torch.Size([1, 4096, 128])
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
            self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        return self.net(hidden_states)


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
            self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64,
            dropout: int = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        return self.to_out(hidden_states)

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                    torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class OffsetRefStrucInter(nn.Module):

    def __init__(
            self,
            res_in_channels: int,
            style_feat_in_channels: int,
            n_heads: int,
            num_groups: int = 64,
            dropout: float = 0.0,
            gated_ff: bool = True,
    ):
        super().__init__()
        # style feat projecter
        self.style_proj_in = nn.Conv2d(style_feat_in_channels, style_feat_in_channels, kernel_size=1, stride=1,
                                       padding=0)
        self.gnorm_s = torch.nn.GroupNorm(num_groups=num_groups, num_channels=style_feat_in_channels, eps=1e-6,
                                          affine=True)
        self.ln_s = nn.LayerNorm(style_feat_in_channels)

        # content feat projecter
        self.content_proj_in = nn.Conv2d(res_in_channels, res_in_channels, kernel_size=1, stride=1, padding=0)
        self.gnorm_c = torch.nn.GroupNorm(num_groups=num_groups, num_channels=res_in_channels, eps=1e-6, affine=True)
        self.ln_c = nn.LayerNorm(res_in_channels)

        # cross-attention
        # dim_head is the middle dealing dimension, output dimension will be change to quert_dim by Linear
        self.cross_attention = CrossAttention(
            query_dim=style_feat_in_channels, context_dim=res_in_channels, heads=n_heads, dim_head=res_in_channels,
            dropout=dropout
        )

        # FFN
        self.ff = FeedForward(style_feat_in_channels, dropout=dropout, glu=gated_ff)
        self.ln_ff = nn.LayerNorm(style_feat_in_channels)

        self.gnorm_out = torch.nn.GroupNorm(num_groups=num_groups, num_channels=style_feat_in_channels, eps=1e-6,
                                            affine=True)
        self.proj_out = nn.Conv2d(style_feat_in_channels, 1 * 2 * 3 * 3, kernel_size=1, stride=1, padding=0)

    def forward(self, res_hidden_states, style_content_hidden_states):
        batch, c_channel, height, width = res_hidden_states.shape
        _, s_channel, _, _ = style_content_hidden_states.shape
        # style projecter
        style_content_hidden_states = self.gnorm_s(style_content_hidden_states)
        style_content_hidden_states = self.style_proj_in(style_content_hidden_states)

        style_content_hidden_states = style_content_hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width,
                                                                                              s_channel)
        style_content_hidden_states = self.ln_s(style_content_hidden_states)

        # content projecter
        res_hidden_states = self.gnorm_c(res_hidden_states)
        res_hidden_states = self.content_proj_in(res_hidden_states)

        res_hidden_states = res_hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, c_channel)
        res_hidden_states = self.ln_c(res_hidden_states)

        # style and content cross-attention
        hidden_states = self.cross_attention(style_content_hidden_states, context=res_hidden_states)

        # ffn
        hidden_states = self.ff(self.ln_ff(hidden_states)) + hidden_states

        # reshape
        _, _, c = hidden_states.shape
        reshape_out = hidden_states.permute(0, 2, 1).reshape(batch, c, height, width)

        # projert out
        reshape_out = self.gnorm_out(reshape_out)
        offset_out = self.proj_out(reshape_out)

        return offset_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


class ChannelAttnBlock(nn.Module):
    """This is the Channel Attention in MCA.
    """

    def __init__(
            self,
            in_channels,  # 输入通道数（即 input 和 content_feature 拼接后的通道总数）
            out_channels,  # 输出通道数（最终的通道维度）
            groups=32,  # GroupNorm 中的组数
            groups_out=None,  # 输出特征的 GroupNorm 的组数（可选，默认和 groups 一致）
            eps=1e-6,  # GroupNorm 中的 epsilon，用于数值稳定性
            non_linearity="swish",  # 激活函数类型（支持 swish、mish、silu）
            channel_attn=False,  # 是否启用通道注意力机制
            reduction=32):  # 如果启用通道注意力，SE模块中的通道压缩比例
        super().__init__()

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.channel_attn = channel_attn
        if self.channel_attn:
            # SE Attention
            self.se_channel_attn = SELayer(channel=in_channels, reduction=reduction)

        # Down channel: Use the conv1*1 to down the channel wise
        self.norm3 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.down_channel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)  # conv1*1

    def forward(self, input, content_feature):
        concat_feature = torch.cat([input, content_feature], dim=1)
        hidden_states = concat_feature

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.channel_attn:
            hidden_states = self.se_channel_attn(hidden_states)
            hidden_states = hidden_states + concat_feature

        # Down channel
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.down_channel(hidden_states)

        return hidden_states


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(
            self,
            in_channels,
            content_channel,
            out_channels,  # 最终输出通道数
            groups=32,
            eps=1e-6,
            reduction=2,
            i=-1
    ):
        super(CAFM, self).__init__()
        self.in_channels = in_channels
        self.content_channel = content_channel
        self.out_channels = out_channels
        self.groups = groups
        self.eps = eps
        self.reduction = reduction
        self.i = i

        self.channels = in_channels + content_channel

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(self.in_channels, self.in_channels // 2, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(self.in_channels, self.in_channels // 2, 1, stride=1, padding=0)
        self.avg11 = nn.Conv2d(self.in_channels // 2, self.in_channels, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(self.in_channels // 2, self.in_channels, 1, stride=1, padding=0)

        self.avg2 = nn.Conv2d(self.content_channel, self.content_channel // 2, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(self.content_channel, self.content_channel // 2, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(self.content_channel // 2, self.content_channel, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(self.content_channel // 2, self.content_channel, 1, stride=1, padding=0)

        self.map_content = nn.Conv2d(self.content_channel, self.in_channels, kernel_size=1)

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=self.channels, eps=eps, affine=True)
        #self.nonlinearity = lambda x: F.silu(x)
        self.nonlinearity = F.silu
        self.down_channel = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels, kernel_size=1)

    def forward(self, input, content_feature):

        b, c, h, w = input.size()
        b1, c1, h1, w1 = content_feature.size()

        input = input.reshape([b, c, -1])
        content_feature = content_feature.reshape([b1, c1, -1])

        avg_1 = torch.mean(input, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(input, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(content_feature, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(content_feature, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2)).detach()

        a1 = torch.matmul(F.softmax(cross, dim=-1), content_feature)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), input)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        with torch.no_grad():
            a1_weight = self.conv2_spatial(F.relu(self.conv1_spatial(torch.cat([avg_out, max_out], dim=1))))
            a1_weight = a1_weight.reshape([b, 1, -1])
            a1_weight = F.softmax(a1_weight, dim=-1)

        a2 = a2.reshape([b1, c1, h1, w1])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        with torch.no_grad():
            a2_weight = self.conv2_spatial(F.relu(self.conv1_spatial(torch.cat([avg_out, max_out], dim=1))))
            a2_weight = a2_weight.reshape([b, 1, -1])
            a2_weight = F.softmax(a2_weight, dim=-1)

        input = input * a1_weight.detach() + input
        content_feature = content_feature * a2_weight.detach() + content_feature

        hidden_states = torch.cat([input, content_feature], dim=1)
        hidden_states = hidden_states.view(b, c + c1, h, w)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.down_channel(hidden_states)
        return hidden_states



class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class PPA(nn.Module):
    def __init__(self, in_features1, in_features2, filters) -> None:
        super().__init__()

        in_features_total = in_features1 + in_features2

        self.skip = conv_block(in_features=in_features_total,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features_total,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)

        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x1, x2):
        # 通道拼接，假设尺寸一致
        x = torch.cat([x1, x2], dim=1)  # dim=1 是通道维度 (B, C1+C2, H, W)

        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)

        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

import math
class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


