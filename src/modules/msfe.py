# -*- coding: utf-8 -*-
"""Multi-Source Style Fusion Encoder (MSFE).

Implementation follows Section 3.1 of the MSCI-Diffusion paper:

    e_hr = E_hr(x_hr),   e_hf = E_hf(x_hf),   e_s = E_s(x_s)          (Eq. 1)

    e_hr_bar = A_hr(e_hr),  e_hf_bar = A_hf(e_hf),
    e_tilde  = sigmoid(e_hr_bar ⊙ e_hf_bar)                           (Eq. 2)

    z_g  = GAP([e_tilde; e_s]) ∈ R^{2Cs}
    α    = sigmoid(W2 δ(W1 z_g + b1) + b2) ∈ (0, 1)                   (Eq. 3)
    e_ms = α e_tilde + (1 - α) e_s

where x_s is the original reference-style image, x_hr its super-resolved
high-resolution view (256x256, obtained from the same reference glyph), and
x_hf its high-pass filtered high-frequency component. A(·) resamples the
spatial dimensions to Hs x Ws and applies a 1x1 convolution to project the
channel dimension to Cs.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .style_encoder import StyleEncoder
from .imdn import IMDN


class HighPassFilter(nn.Module):
    """Fixed (non-learnable) Laplacian high-pass filter used to derive the
    high-frequency component x_hf from the reference glyph x_s. The filter
    emphasises rapid intensity variations along stroke contours/boundaries.
    """

    def __init__(self, channels: int = 3, kernel_size: int = 3):
        super().__init__()
        assert kernel_size == 3, "Only the 3x3 Laplacian kernel is supported."
        kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0, 1.0, 0.0]])
        weight = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1, groups=self.channels)


class AlignmentOperator(nn.Module):
    """Branch-specific alignment operator A(·) in Eq. (2): resample the
    spatial dimensions to (Hs, Ws) and apply a 1x1 convolution to project the
    channel dimension to Cs."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        if x.shape[-2:] != tuple(out_size):
            x = F.interpolate(x, size=tuple(out_size), mode="bilinear", align_corners=False)
        return self.conv(x)


class MultiSourceStyleFusionEncoder(nn.Module):
    """MSFE: three parallel style branches (original / high-resolution /
    high-frequency) with alignment, structure-enhanced modulation and the
    adaptive gated fusion of Eq. (3). The output e_ms is a feature map of
    shape (B, Cs, Hs, Ws) which serves as the key/value source of the HSCIM
    cross-attention.
    """

    def __init__(
            self,
            G_ch: int = 64,
            G_wide: bool = True,
            resolution: int = 128,
            G_kernel_size: int = 3,
            G_attn: str = "64_32_16_8",
            n_classes: int = 1000,
            num_G_SVs: int = 1,
            num_G_SV_itrs: int = 1,
            G_activation=None,
            SN_eps: float = 1e-12,
            output_dim: int = 1,
            G_fp16: bool = False,
            G_init: str = "N02",
            G_param: str = "SN",
            nf_mlp: int = 512,
            nEmbedding: int = 256,
            input_nc: int = 3,
            output_nc: int = 3,
            upscale: int = 2,
            imdn_ckpt_path: str = "ckpt/IMDN_x2.pth",
            gate_hidden_dim: int = None,
    ):
        super().__init__()
        if G_activation is None:
            G_activation = nn.ReLU(inplace=False)

        self.resolution = resolution
        self.upscale = upscale

        # ------------------------------------------------------------------
        # Three parallel branches (Eq. 1)
        # ------------------------------------------------------------------
        # Style encoder on the original reference-style image x_s.
        self.style_encoder = StyleEncoder(
            G_ch=G_ch, G_wide=G_wide, resolution=resolution,
            G_kernel_size=G_kernel_size, G_attn=G_attn, n_classes=n_classes,
            num_G_SVs=num_G_SVs, num_G_SV_itrs=num_G_SV_itrs,
            G_activation=G_activation, SN_eps=SN_eps, output_dim=output_dim,
            G_fp16=G_fp16, G_init=G_init, G_param=G_param, nf_mlp=nf_mlp,
            nEmbedding=nEmbedding, input_nc=input_nc, output_nc=output_nc)
        # High-resolution encoder on the super-resolved view x_hr.
        self.hr_encoder = StyleEncoder(
            G_ch=G_ch, G_wide=G_wide, resolution=resolution * upscale,
            G_kernel_size=G_kernel_size, G_attn=G_attn, n_classes=n_classes,
            num_G_SVs=num_G_SVs, num_G_SV_itrs=num_G_SV_itrs,
            G_activation=G_activation, SN_eps=SN_eps, output_dim=output_dim,
            G_fp16=G_fp16, G_init=G_init, G_param=G_param, nf_mlp=nf_mlp,
            nEmbedding=nEmbedding, input_nc=input_nc, output_nc=output_nc)
        # High-frequency encoder on the high-pass filtered image x_hf.
        self.hf_encoder = StyleEncoder(
            G_ch=G_ch, G_wide=G_wide, resolution=resolution,
            G_kernel_size=G_kernel_size, G_attn=G_attn, n_classes=n_classes,
            num_G_SVs=num_G_SVs, num_G_SV_itrs=num_G_SV_itrs,
            G_activation=G_activation, SN_eps=SN_eps, output_dim=output_dim,
            G_fp16=G_fp16, G_init=G_init, G_param=G_param, nf_mlp=nf_mlp,
            nEmbedding=nEmbedding, input_nc=input_nc, output_nc=output_nc)

        # Channel dimension Cs of the style feature e_s and (Hs, Ws).
        self.style_channels = self.style_encoder.arch["out_channels"][-1]
        hr_channels = self.hr_encoder.arch["out_channels"][-1]
        hf_channels = self.hf_encoder.arch["out_channels"][-1]

        # ------------------------------------------------------------------
        # High-resolution view: super-resolution of the same reference glyph
        # (Section 4.2.1). A pretrained IMDN is used when its checkpoint is
        # available; otherwise a (parameter-free) bilinear upsampling is used
        # so the model stays trainable out of the box.
        # ------------------------------------------------------------------
        self.imdn_ckpt_path = imdn_ckpt_path
        self.sr_net = None
        try:
            sr_net = IMDN(upscale=self.upscale)
            state_dict = torch.load(imdn_ckpt_path, map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            sr_net.load_state_dict(new_state_dict)
            self.sr_net = sr_net
            print(f"MSFE: loaded the pretrained SR model from {imdn_ckpt_path}.")
        except Exception as e:  # noqa: BLE001
            print(f"MSFE: SR checkpoint not loaded ({e}). "
                  f"Falling back to bilinear x{self.upscale} upsampling for the "
                  f"high-resolution view. Place the pretrained SR weights at "
                  f"'{imdn_ckpt_path}' to enable super-resolution.")
        if self.sr_net is not None:
            self.sr_net.requires_grad_(False)
            self.sr_net.eval()

        # Fixed high-pass filter for the high-frequency view x_hf.
        self.high_pass = HighPassFilter(channels=input_nc)

        # ------------------------------------------------------------------
        # Alignment operators A_hr / A_hf (Eq. 2)
        # ------------------------------------------------------------------
        self.align_hr = AlignmentOperator(hr_channels, self.style_channels)
        self.align_hf = AlignmentOperator(hf_channels, self.style_channels)

        # ------------------------------------------------------------------
        # Adaptive gating (Eq. 3). W1 ∈ R^{dg x 2Cs}, W2 ∈ R^{1 x dg}; the
        # scalar α is computed independently for each sample and broadcast
        # over channel and spatial dimensions.
        # ------------------------------------------------------------------
        gate_hidden_dim = gate_hidden_dim or max(self.style_channels // 4, 8)
        self.gate_fc1 = nn.Linear(2 * self.style_channels, gate_hidden_dim)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, 1)

    def super_resolution(self, style_img: torch.Tensor) -> torch.Tensor:
        """Obtain the high-resolution view x_hr from the reference glyph."""
        if self.sr_net is not None:
            with torch.no_grad():
                return self.sr_net(style_img)
        return F.interpolate(style_img, scale_factor=self.upscale,
                             mode="bilinear", align_corners=False)

    def forward(self, style_img: torch.Tensor) -> torch.Tensor:
        # Derive the three views of the same reference glyph.
        x_s = style_img
        x_hr = self.super_resolution(x_s)
        x_hf = self.high_pass(x_s)

        # Eq. (1): three parallel branches.
        e_s = self.style_encoder(x_s)[0]
        e_hr = self.hr_encoder(x_hr)[0]
        e_hf = self.hf_encoder(x_hf)[0]

        # Eq. (2): align to the common space of e_s, then Hadamard modulation.
        Hs, Ws = e_s.shape[-2:]
        e_hr_bar = self.align_hr(e_hr, (Hs, Ws))
        e_hf_bar = self.align_hf(e_hf, (Hs, Ws))
        e_tilde = torch.sigmoid(e_hr_bar * e_hf_bar)

        # Eq. (3): global-average-pooled gating and fusion.
        z_g = torch.cat([e_tilde.mean(dim=(2, 3)), e_s.mean(dim=(2, 3))], dim=1)
        alpha = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(z_g))))  # (B, 1)
        e_ms = alpha[..., None, None] * e_tilde + (1.0 - alpha[..., None, None]) * e_s
        return e_ms
