# -*- coding: utf-8 -*-
"""Hierarchical Style-Content Interaction Module (HSCIM).

Implementation follows Section 3.2 of the MSCI-Diffusion paper. At each
resolution level i of the denoising U-Net decoder, the module receives:

  - the content feature  e_c^i ∈ R^{B x Cc x Hi x Wi} (content encoder),
  - the denoising feature r_i   ∈ R^{B x Cr x Hi x Wi} propagated from the
    preceding U-Net block,
  - the multi-source style representation e_ms ∈ R^{B x Cs x Hs x Ws}.

and performs the cascaded interaction:

  1) Channel attention with bidirectional cross-channel correlation built by
     outer products and row-wise Softmax (Eq. 4-6).
  2) Spatial attention recalibrating the channel-interacted features
     (Eq. 7-9).
  3) Channel concatenation followed by GroupNorm, non-linearity and a 1x1
     convolution producing F_down (the paragraph after Eq. 10).
  4) Cross-attention that uses e_ms as key/value to guide the fused content
     feature (Eq. 11-13).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_num_groups(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of `channels` not greater than `max_groups`."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ChannelDescriptorMLP(nn.Module):
    """Shared channel-reduction / channel-expansion projection of Eq. (5):
    Convexp(δ(Convred(·))). The two pooling branches (avg / max) share the
    projection parameters."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: (B, C)
        return self.expand(F.relu(self.reduce(pooled)))


class SpatialAttention(nn.Module):
    """Spatial attention of Eq. (7)-(8):
    P = [AvgPool(F), MaxPool(F)];  S = sigmoid(Conv2(δ(Conv1(P))))."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([torch.mean(x, dim=1, keepdim=True),
                            torch.amax(x, dim=1, keepdim=True)], dim=1)
        return self.sigmoid(self.conv2(F.relu(self.conv1(pooled))))  # (B,1,H,W)


class StyleContentCrossAttention(nn.Module):
    """Cross-attention of Eq. (11)-(13). The locally fused feature F_down is
    flattened as the query; the style representation e_ms is flattened as
    key/value. W_Q ∈ R^{Cr x dk}, W_K ∈ R^{Cs x dk}, W_V ∈ R^{Cs x Cr}."""

    def __init__(self, query_channels: int, keyvalue_channels: int):
        super().__init__()
        self.query_channels = query_channels
        self.dk = query_channels  # single-head attention, dim_head = Cr
        self.to_q = nn.Linear(query_channels, self.dk, bias=False)      # W_Q
        self.to_k = nn.Linear(keyvalue_channels, self.dk, bias=False)   # W_K
        self.to_v = nn.Linear(keyvalue_channels, query_channels, bias=False)  # W_V
        self.scale = self.dk ** -0.5

    def forward(self, f_down: torch.Tensor, e_ms: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = f_down.shape
        # Flat(·): collapse spatial dims and transpose to token sequences.
        x_q = f_down.flatten(2).transpose(1, 2)   # (B, Nq, Cr)
        x_s = e_ms.flatten(2).transpose(1, 2)     # (B, Ns, Cs)

        query = self.to_q(x_q)                    # (B, Nq, dk)
        key = self.to_k(x_s)                      # (B, Ns, dk)
        value = self.to_v(x_s)                    # (B, Ns, Cr)

        attn = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale,
                             dim=-1)              # (B, Nq, Ns), softmax over Ns
        out = torch.matmul(attn, value)           # (B, Nq, Cr)
        # Unflat(·): restore the query spatial arrangement.
        return out.transpose(1, 2).reshape(batch, self.query_channels, height, width)


class HSCIM(nn.Module):
    """Hierarchical Style-Content Interaction Module.

    Args:
        unet_channels: Cr, channels of the denoising feature r_i.
        content_channels: Cc, channels of the content feature e_c^i.
        style_channels: Cs, channels of the multi-source style feature e_ms.
        reduction: channel reduction ratio of the shared channel MLP.
    """

    def __init__(
            self,
            unet_channels: int,
            content_channels: int,
            style_channels: int,
            reduction: int = 32,
    ):
        super().__init__()
        self.unet_channels = unet_channels
        self.content_channels = content_channels

        # Channel attention (Eq. 4-5), one shared MLP per branch.
        self.channel_mlp_r = ChannelDescriptorMLP(unet_channels, reduction)
        self.channel_mlp_c = ChannelDescriptorMLP(content_channels, reduction)

        # Spatial attention (Eq. 7-8), one per branch.
        self.spatial_r = SpatialAttention()
        self.spatial_c = SpatialAttention()

        # Channel fusion: GroupNorm + non-linearity + 1x1 conv (after Eq. 10).
        fuse_channels = unet_channels + content_channels
        self.fuse_norm = nn.GroupNorm(_pick_num_groups(fuse_channels), fuse_channels, eps=1e-6, affine=True)
        self.fuse_conv = nn.Conv2d(fuse_channels, unet_channels, kernel_size=1, stride=1, padding=0)

        # Style-guided cross-attention (Eq. 11-13).
        self.cross_attn = StyleContentCrossAttention(unet_channels, style_channels)

    @staticmethod
    def _channel_interaction(feature: torch.Tensor, mlp: ChannelDescriptorMLP) -> torch.Tensor:
        """Channel descriptor a = Convexp(δ(Convred(avg))) + Convexp(δ(Convred(max)))."""
        avg = torch.mean(feature, dim=(2, 3))
        mx = torch.amax(feature, dim=(2, 3))
        return mlp(avg) + mlp(mx)  # (B, C)

    def forward(self, r: torch.Tensor, c: torch.Tensor, e_ms: torch.Tensor) -> torch.Tensor:
        batch, cr, height, width = r.shape
        cc = c.shape[1]

        # Align the content feature to the denoising feature resolution (Hi x Wi).
        if c.shape[-2:] != (height, width):
            c = F.interpolate(c, size=(height, width), mode="bilinear", align_corners=False)

        # ------------------------------------------------------------------
        # 1) Channel attention + bidirectional cross-channel correlation
        #    (Eq. 4-6)
        # ------------------------------------------------------------------
        a_r = self._channel_interaction(r, self.channel_mlp_r)   # (B, Cr)
        a_c = self._channel_interaction(c, self.channel_mlp_c)   # (B, Cc)

        x_r = r.flatten(2)   # (B, Cr, N)
        x_c = c.flatten(2)   # (B, Cc, N)

        # Outer products (NOT Hadamard): M_rc ∈ R^{Cr x Cc}, M_cr ∈ R^{Cc x Cr}.
        m_rc = torch.matmul(a_r.unsqueeze(2), a_c.unsqueeze(1))
        m_cr = torch.matmul(a_c.unsqueeze(2), a_r.unsqueeze(1))
        # Row-wise Softmax normalisation.
        m_rc = torch.softmax(m_rc, dim=-1)
        m_cr = torch.softmax(m_cr, dim=-1)

        f_r_ca = torch.matmul(m_rc, x_c).reshape(batch, cr, height, width)   # F_r^ca
        f_c_ca = torch.matmul(m_cr, x_r).reshape(batch, cc, height, width)   # F_c^ca

        # ------------------------------------------------------------------
        # 2) Spatial attention recalibration (Eq. 7-9)
        # ------------------------------------------------------------------
        s_r = self.spatial_r(f_r_ca)   # (B, 1, H, W)
        s_c = self.spatial_c(f_c_ca)   # (B, 1, H, W)
        f_r_prime = f_r_ca * s_r       # broadcast over channels
        f_c_prime = f_c_ca * s_c

        # ------------------------------------------------------------------
        # 3) Channel concatenation + GroupNorm + act + 1x1 conv → F_down
        # ------------------------------------------------------------------
        f_cat = torch.cat([f_r_prime, f_c_prime], dim=1)      # (B, Cr+Cc, H, W)
        f_down = self.fuse_conv(F.silu(self.fuse_norm(f_cat)))  # (B, Cr, H, W)

        # ------------------------------------------------------------------
        # 4) Style-guided cross-attention with e_ms as key / value
        #    (Eq. 11-13)
        # ------------------------------------------------------------------
        f_out = self.cross_attn(f_down, e_ms)                 # (B, Cr, H, W)
        return f_out
