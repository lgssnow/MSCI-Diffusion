import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin,
                                           register_to_config)


class MSCIDiffusionModel(ModelMixin, ConfigMixin):
    """MSCI-Diffusion model (paper Section 3):
        - MultiSourceStyleFusionEncoder (MSFE) -> e_ms,
        - ContentEncoder -> multi-resolution content pyramid {e_c^i},
        - conditional diffusion U-Net with HSCIM at every decoder scale.

    The content encoder is used only to extract the structure of the source
    glyph and does not receive the target-style reference (Section 3.3).
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
    ):
        # MSFE: multi-source style representation e_ms, (B, Cs, Hs, Ws)
        style_img_feature = self.style_encoder(style_images)

        # Content encoder: multi-resolution content-feature pyramid {e_c^i}
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)

        noise_pred = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=[style_img_feature, content_residual_features],
            content_encoder_downsample_size=content_encoder_downsample_size,
        )

        # MSCI-Diffusion has no SRIC offset branch; keep the interface of the
        # original training loop (offset term is always zero).
        return noise_pred, torch.zeros((), device=x_t.device)


class MSCIDiffusionModelDPM(ModelMixin, ConfigMixin):
    """DPM-solver forward function used at inference time."""
    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        cond,  # 条件输入，(content_images, style_images)
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]

        # MSFE: e_ms
        style_img_feature = self.style_encoder(style_images)

        # Content pyramid of the source glyph
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)

        noise_pred = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=[style_img_feature, content_residual_features],
            content_encoder_downsample_size=content_encoder_downsample_size,
        )

        return noise_pred
