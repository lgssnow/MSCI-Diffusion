import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import pandas as pd

# === 指标计算函数 ===

def compute_l1(img1, img2):
    return F.l1_loss(img1, img2).item()

def compute_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, data_range=1.0, multichannel=True)

def compute_lpips(img1, img2, loss_fn):
    return loss_fn(img1, img2).item()

# === 图片预处理 ===

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

to_image = transforms.Compose([
    transforms.Normalize([-1]*3, [2]*3),
    transforms.ToPILImage()
])

# === 主测试函数 ===

def evaluate_dataset(pipe, args):
    content_root = 'content'
    style_root = 'style'
    output_root = 'sample'
    device = args.device
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    metrics = []

    for style_name in os.listdir(style_root):
        style_dir = os.path.join(style_root, style_name)
        save_dir = os.path.join(output_root, style_name)
        os.makedirs(save_dir, exist_ok=True)

        for content_name in os.listdir(content_root):
            content_id = os.path.splitext(content_name)[0]
            style_path = os.path.join(style_dir, f"{style_name}+{content_id}.png")
            content_path = os.path.join(content_root, content_name)
            output_path = os.path.join(save_dir, f"{content_id}.png")

            # 加载图像
            content_img = Image.open(content_path).convert("RGB")
            style_img = Image.open(style_path).convert("RGB")

            content_tensor = to_tensor(content_img).unsqueeze(0).to(device)
            style_tensor = to_tensor(style_img).unsqueeze(0).to(device)

            # 生成图像
            with torch.no_grad():
                gen_img_tensor = pipe.generate(
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

            # 保存图像
            to_image(gen_img_tensor.cpu()).save(output_path)

            # 指标计算
            gt_tensor = style_tensor
            lp = compute_lpips(gen_img_tensor, gt_tensor, loss_fn_lpips)
            l1 = compute_l1(gen_img_tensor, gt_tensor)
            ssim_score = compute_ssim(gen_img_tensor[0], gt_tensor[0])

            metrics.append([style_name, None, ssim_score, lp, l1])  # FID 需批量统一计算

    # 保存表格
    df = pd.DataFrame(metrics, columns=["风格", "FID", "SSIM", "LPIPS", "L1"])
    df.to_excel("eval_results.xlsx", index=False)
    print("评估结果已保存为 eval_results.xlsx")

