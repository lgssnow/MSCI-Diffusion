<div align="center">

# MSCI-Diffusion

**Multi-Source Style-Content Interaction for Few-Shot Font Generation**

A conditional diffusion framework for few-shot Chinese font generation.

</div>

## 📝 Introduction

MSCI-Diffusion synthesizes target-style character sets from a single reference glyph while preserving the structure of the source character. The framework consists of three components:

1. **Multi-Source Style Fusion Encoder (MSFE)** — adaptively fuses style cues from three views derived from the same reference glyph: the original reference image, its high-resolution view (×2 super-resolution), and its high-frequency component (fixed Laplacian high-pass filtering). The aligned features are combined by an element-wise modulation and a per-sample gated fusion into a unified style representation `e_ms`.

2. **Hierarchical Style-Content Interaction Module (HSCIM)** — coordinates the style and content features at every resolution scale of the denoising U-Net decoder. At each scale, the denoising feature `r_i` and the content feature `e_c^i` pass through channel attention (avg/max descriptors with bidirectional cross-channel correlation), spatial attention, channel fusion (GroupNorm + activation + 1×1 conv), and finally a style-guided cross-attention that takes `e_ms` as key/value — realizing coarse-to-fine style injection.

3. **Conditional Diffusion Generator** — a DDPM-based U-Net (1000 training timesteps) trained with the noise-prediction MSE objective. Sampling uses the DDIM sampler with 50 denoising steps.

## 🔧 Environment

```bash
conda create -n msci python=3.10 -y
conda activate msci
pip install torch torchvision  # CUDA build on GPU machines
pip install -r requirements.txt
```

## 📁 Dataset Format

```
data_root/
└── train/
    ├── ContentImage/          # source-font glyphs:  {char}.png
    │   ├── 永.png
    │   └── ...
    └── TargetImage/           # target-font glyphs:  {style}+{char}.png
        ├── styleA/
        │   ├── styleA+永.png
        │   └── ...
        └── ...
```

All glyph images are RGB PNGs (rendered at 128×128 recommended).

## 🏋️ Training

Single-stage training (paper Section 4.2.2):

```bash
bash scripts/train.sh
```

Key settings (paper Section 4.2): 128×128 resolution, batch size 16, Adam optimizer with lr 1e-4, linear warmup of 10,000 steps, 440,000 total iterations. Checkpoints are saved under `outputs/MSCI_Diffusion/global_step_{N}/` (`unet.pth`, `style_encoder.pth`, `content_encoder.pth`).

**High-resolution branch (optional but recommended)**: place a pretrained ×2 super-resolution checkpoint at `ckpt/IMDN_x2.pth`. If the file is absent, the model falls back to bilinear ×2 upsampling automatically.

## 📺 Sampling / Evaluation

```bash
python test.py \
    --ckpt_dir outputs/MSCI_Diffusion/global_step_440000 \
    --content_image_path data_examples/sampling/example_content_1.png \
    --style_image_path data_examples/sampling/example_style.jpg \
    --save_image --save_image_dir outputs/sampled \
    --resolution=128 --style_image_size=128 --content_image_size=128
```

Default sampler: DDIM, 50 denoising steps (implemented with the first-order DPM-Solver, mathematically equivalent to DDIM).

## ✅ Self-check Scripts

```bash
python smoke_test.py            # build + forward + backward + optimizer step
python verify_paper.py          # 28 empirical checks of every model equation
```

## 📐 Repository Layout

```
├── train.py                    # training loop (diffusion MSE objective)
├── test.py                     # batch generation on a test set
├── sample.py                   # single-image sampling
├── configs/msci_diffusion.py   # all training / model arguments
├── dataset/                    # dataset & collate
├── scripts/                    # launch scripts
└── src/
    ├── model.py                # MSCIDiffusionModel / DPM variant
    ├── build.py                # model & scheduler builders
    ├── criterion.py            # optional perceptual loss
    ├── dpm_solver/             # sampling pipeline
    └── modules/
        ├── msfe.py             # Multi-Source Style Fusion Encoder
        ├── hscim.py            # Hierarchical Style-Content Interaction Module
        ├── unet.py             # conditional diffusion U-Net
        ├── unet_blocks.py      # down / mid / HSCIM up blocks
        ├── content_encoder.py  # content feature pyramid
        └── ...
```
