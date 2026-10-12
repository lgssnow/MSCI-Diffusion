# -*- coding: utf-8 -*-
"""Smoke test: build the MSCI-Diffusion model exactly as in training and run a
forward / backward / optimizer step on the CPU.

Usage:
    python smoke_test.py                 # paper config (128x128, batch 1)
    python smoke_test.py --size 96 --batch 2
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imdn_ckpt_path", type=str, default="ckpt/IMDN_x2.pth")
    args_cli = parser.parse_args()

    from configs.msci_diffusion import get_parser
    from src import (MSCIDiffusionModel,
                     build_unet,
                     build_style_encoder,
                     build_content_encoder,
                     build_ddpm_scheduler)

    parser = get_parser()
    args = parser.parse_args([])
    args.resolution = args_cli.size
    args.style_image_size = (args_cli.size, args_cli.size)
    args.content_image_size = (args_cli.size, args_cli.size)
    args.imdn_ckpt_path = args_cli.imdn_ckpt_path

    device = torch.device("cpu")

    print("==> building model ...")
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    scheduler = build_ddpm_scheduler(args)

    model = MSCIDiffusionModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    trainable parameters: {n_params / 1e6:.2f} M")

    b = args_cli.batch
    content_images = torch.rand(b, 3, args_cli.size, args_cli.size, device=device) * 2 - 1
    style_images = torch.rand(b, 3, args_cli.size, args_cli.size, device=device) * 2 - 1
    target_images = torch.rand(b, 3, args_cli.size, args_cli.size, device=device) * 2 - 1

    e_ms = model.style_encoder(style_images)
    print(f"    MSFE output e_ms: {tuple(e_ms.shape)}")

    noise = torch.randn_like(target_images)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (b,), device=device).long()
    x_t = scheduler.add_noise(target_images, noise, timesteps)

    print("==> forward ...")
    noise_pred, offset = model(
        x_t=x_t,
        timesteps=timesteps,
        style_images=style_images,
        content_images=content_images,
        content_encoder_downsample_size=args.content_encoder_downsample_size)
    print(f"    noise_pred: {tuple(noise_pred.shape)}, offset term: {float(offset)}")

    print("==> backward + optimizer step ...")
    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
    loss.backward()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()
    print(f"    diff loss: {loss.item():.6f}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
