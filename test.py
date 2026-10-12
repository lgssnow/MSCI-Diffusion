import csv
import os
import shutil
import time
from tqdm import tqdm

import pandas as pd
import torch
import random
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from pytorch_fid.fid_score import calculate_fid_given_paths

# 导入你原来定义的相关函数与模型构建函数
from src import (MSCIDiffusionDPMPipeline,
                 MSCIDiffusionModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)


def arg_parse():
    from configs.msci_diffusion import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False,
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def image_process_for_single(content_image, style_image, args):
    """对单个 PIL 图像进行 transform """
    content_inference_transforms = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_inference_transforms = transforms.Compose([
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    content_tensor = content_inference_transforms(content_image)[None, :]
    style_tensor = style_inference_transforms(style_image)[None, :]
    return content_tensor, style_tensor


def load_pipeline(args):
    # # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = MSCIDiffusionModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    #model = torch.load(f"{args.ckpt_dir}/total_model.pth", map_location=args.device)
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler successfully!")

    # Load the DPM_Solver pipeline.
    pipe = MSCIDiffusionDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline successfully!")

    return pipe




def evaluate_dataset(args, pipe):
    """
    遍历测试数据集，对每个风格文件夹下所有内容图生成结果进行指标评估。
    数据集格式：
    data_examples/sample/
      ├── ContentImage/         # 含有 char0.png, char1.png, ...
      └── TargetImage/          # 子文件夹每个风格，如 style0, style1, ...
             ├── style0+char0.png
             ├── style0+char1.png
             └── ...
    """

    # 设置路径（根据你的数据集实际位置修改）
    content_dir = "data_examples/test/Content"
    target_dir = "data_examples/test/Target"  # 每个子文件夹为一种风格

    # 生成结果保存目录（用于 FID 计算）
    gen_root = "output_generated"
    os.makedirs(gen_root, exist_ok=True)
    # 遍历所有风格文件夹
    style_folders = [os.path.join(target_dir, d) for d in os.listdir(target_dir)
                     if os.path.isdir(os.path.join(target_dir, d))]
    for style_folder in style_folders:
        style_name = os.path.basename(style_folder)
        # 选择该风格文件夹中的一个代表性风格图（例如，第一个 .png 文件）
        style_files = [f for f in os.listdir(style_folder) if f.endswith('.png')]
        if not style_files:
            continue
        style_image_path = os.path.join(style_folder, style_files[0])
        style_image_ref = Image.open(style_image_path).convert('RGB')

        # 为生成结果建立当前风格的输出文件夹
        gen_style_dir = os.path.join(gen_root, style_name)
        os.makedirs(gen_style_dir, exist_ok=True)

        # 遍历 ContentImage 下的所有内容图
        content_files = [f for f in os.listdir(content_dir) if f.endswith('.png')]
        for content_file in tqdm(content_files, desc=f"Processing {style_name}", unit="img"):
            content_image_path = os.path.join(content_dir, content_file)
            content_image = Image.open(content_image_path).convert('RGB')
            # 使用同一个代表性风格图作为输入 style
            content_tensor, style_tensor = image_process_for_single(content_image, style_image_ref, args)
            content_tensor = content_tensor.to(args.device)
            style_tensor = style_tensor.to(args.device)

            with torch.no_grad():
                # 生成图像，此处假设返回的是 PIL Image
                gen_image = pipe.generate(
                    content_images=content_tensor,
                    style_images=style_tensor,
                    batch_size=1,
                    order=args.order,
                    num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn
                )[0]

            # 若 pipeline 返回的是 PIL Image，则可直接使用；否则需调整
            # 如需将其归一化处理，可先转换为 tensor，再转换回 PIL Image
            # to_tensor = transforms.ToTensor()
            # gen_tensor = to_tensor(gen_image).to(torch.float32)
            # gen_tensor = (gen_tensor * 0.5 + 0.5).clamp(0, 1)
            # gen_image = transforms.ToPILImage()(gen_tensor.cpu())

            # 保存生成图像，命名格式：风格名+{内容图名（不含扩展名）}.png
            gen_filename = f"{style_name}+{os.path.splitext(content_file)[0]}.png"
            gen_save_path = os.path.join(gen_style_dir, gen_filename)
            gen_image.save(gen_save_path)







if __name__ == "__main__":
    args = arg_parse()

    # 加载 pipeline
    pipe = load_pipeline(args=args)

    print("Start evaluating the test dataset ...")
    evaluate_dataset(args, pipe)
    print("Evaluation finished.")
