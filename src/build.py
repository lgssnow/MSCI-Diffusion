from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src import (ContentEncoder,
                 UNet)
from src.modules.msfe import MultiSourceStyleFusionEncoder


def build_unet(args):
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=('DownBlock2D',
                          'DownBlock2D',
                          'DownBlock2D',
                          'DownBlock2D'),
        up_block_types=('HSCIMUpBlock2D',
                        'HSCIMUpBlock2D',
                        'HSCIMUpBlock2D',
                        'HSCIMUpBlock2D'),
        block_out_channels=args.unet_channels,
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        content_resolution=args.content_image_size[0],
        reduction=32)

    return unet


def build_style_encoder(args):
    """Build the Multi-Source Style Fusion Encoder (MSFE)."""
    style_image_encoder = MultiSourceStyleFusionEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size[0],
        imdn_ckpt_path=args.imdn_ckpt_path)
    print("Get MSCI-Diffusion Multi-Source Style Fusion Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size[0])
    print("Get MSCI-Diffusion Content Encoder!")
    return content_image_encoder


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True)
    return ddpm_scheduler
