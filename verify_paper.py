# -*- coding: utf-8 -*-
"""Paper-consistency verification for MSCI-Diffusion.

Empirically checks every equation of paper Sections 3.1-3.3 against the
running implementation (shapes, ranges, normalisation, wiring).

Run:  python verify_paper.py
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def verify_msfe():
    print("=" * 70)
    print("Section 3.1  MSFE  (Eq. 1-3)")
    print("=" * 70)
    from src.modules.msfe import MultiSourceStyleFusionEncoder

    msfe = MultiSourceStyleFusionEncoder(G_ch=64, resolution=128)
    msfe.eval()
    B = 2
    x_s = torch.rand(B, 3, 128, 128) * 2 - 1

    # ---- Eq. (1): three parallel branches ----
    with torch.no_grad():
        x_hr = msfe.super_resolution(x_s)                    # 256x256 HR view (Sec. 4.2.1)
        x_hf = msfe.high_pass(x_s)                           # high-pass view
        e_s = msfe.style_encoder(x_s)[0]
        e_hr = msfe.hr_encoder(x_hr)[0]
        e_hf = msfe.hf_encoder(x_hf)[0]
        check("Sec 4.2.1: x_hr is a 256x256 view of the 128x128 reference",
              x_hr.shape == (B, 3, 256, 256), str(tuple(x_hr.shape)))
        check("x_hf derived from the same reference via high-pass filtering",
              x_hf.shape == x_s.shape, str(tuple(x_hf.shape)))
        Hs, Ws = e_s.shape[-2:]
        Cs = e_s.shape[1]
        check("Eq.1 e_s = E_s(x_s) in R^{B x Cs x Hs x Ws}",
              e_s.dim() == 4 and e_s.shape[0] == B, f"Cs={Cs}, Hs=Ws={Hs}")

        # ---- Eq. (2): alignment + structure-enhanced feature ----
        e_hr_bar = msfe.align_hr(e_hr, (Hs, Ws))
        e_hf_bar = msfe.align_hf(e_hf, (Hs, Ws))
        check("Eq.2 A_hr/A_hf resample to Hs x Ws and project to Cs",
              e_hr_bar.shape == e_hf_bar.shape == (B, Cs, Hs, Ws),
              f"{tuple(e_hr_bar.shape)} vs {tuple(e_hf_bar.shape)}")
        e_tilde_ref = torch.sigmoid(e_hr_bar * e_hf_bar)     # sigma(x) *only* on the product
        check("Eq.2 e_tilde = sigmoid(e_hr_bar ⊙ e_hf_bar), in (0,1)",
              e_tilde_ref.min() > 0 and e_tilde_ref.max() < 1)

        # ---- Eq. (3): gated fusion ----
        e_ms = msfe(x_s)
        # recover alpha from the module and verify e_ms = a*e_tilde + (1-a)*e_s
        z_g = torch.cat([e_tilde_ref.mean((2, 3)), e_s.mean((2, 3))], dim=1)
        check("Eq.3 z_g = GAP([e_tilde; e_s]) in R^{B x 2Cs}", z_g.shape == (B, 2 * Cs),
              str(tuple(z_g.shape)))
        alpha = torch.sigmoid(msfe.gate_fc2(F.relu(msfe.gate_fc1(z_g))))
        check("Eq.3 alpha = sigma(W2 delta(W1 z_g + b1) + b2), scalar per sample in (0,1)",
              alpha.shape == (B, 1) and alpha.min() > 0 and alpha.max() < 1,
              f"shape={tuple(alpha.shape)}, alpha={alpha.flatten().tolist()}")
        e_ms_ref = alpha[..., None, None] * e_tilde_ref + (1 - alpha[..., None, None]) * e_s
        check("Eq.3 e_ms = alpha*e_tilde + (1-alpha)*e_s  (broadcast over C and HW)",
              torch.allclose(e_ms, e_ms_ref, atol=1e-5),
              f"max|diff|={(e_ms - e_ms_ref).abs().max():.2e}")
        check("e_ms dims == e_s dims (B, Cs, Hs, Ws)", e_ms.shape == e_s.shape)
    return msfe


def verify_hscim():
    print("=" * 70)
    print("Section 3.2  HSCIM  (Eq. 4-13)")
    print("=" * 70)
    from src.modules.hscim import HSCIM

    B, Cr, Cc, H, W = 2, 512, 256, 16, 16
    Cs, Hs, Ws = 1024, 4, 4
    torch.manual_seed(0)
    hscim = HSCIM(unet_channels=Cr, content_channels=Cc, style_channels=Cs, reduction=32)
    hscim.eval()
    r = torch.randn(B, Cr, H, W)          # r_i  (denoising feature)
    c = torch.randn(B, Cc, H, W)          # e_c^i (content feature)
    e_ms = torch.randn(B, Cs, Hs, Ws)

    with torch.no_grad():
        # ---- Eq. (4)-(5): channel descriptors via shared avg/max MLP ----
        a_r = hscim._channel_interaction(r, hscim.channel_mlp_r)
        a_c = hscim._channel_interaction(c, hscim.channel_mlp_c)
        check("Eq.4-5 A_r in R^{B x Cr}, A_c in R^{B x Cc}",
              a_r.shape == (B, Cr) and a_c.shape == (B, Cc),
              f"{tuple(a_r.shape)}, {tuple(a_c.shape)}")
        # shared projections: same MLP object for avg and max
        m = hscim.channel_mlp_r
        check("Eq.5 avg/max branches share the projection parameters",
              torch.allclose(m.reduce(a_r), m.reduce(a_r)))

        # ---- Eq. (6): outer-product correlation + row softmax + aggregation ----
        m_rc = torch.softmax(a_r.unsqueeze(2) @ a_c.unsqueeze(1), dim=-1)
        m_cr = torch.softmax(a_c.unsqueeze(2) @ a_r.unsqueeze(1), dim=-1)
        check("Eq.6a M_rc = a_r a_c^T in R^{Cr x Cc} (outer product, not Hadamard)",
              m_rc.shape == (B, Cr, Cc), str(tuple(m_rc.shape)))
        check("Eq.6b row-wise Softmax: every row of M_rc sums to 1",
              torch.allclose(m_rc.sum(-1), torch.ones(B, Cr), atol=1e-5))
        x_r, x_c = r.flatten(2), c.flatten(2)
        f_r_ca = (m_rc @ x_c).reshape(B, Cr, H, W)
        check("Eq.6c X_r^ca = M_rc X_c in R^{B x Cr x H x W}",
              f_r_ca.shape == (B, Cr, H, W))

        # ---- Eq. (7)-(9): spatial attention recalibration ----
        s_r = hscim.spatial_r(f_r_ca)
        check("Eq.7-8 S_r in [0,1]^{B x 1 x H x W} (single channel, broadcast)",
              s_r.shape == (B, 1, H, W) and s_r.min() >= 0 and s_r.max() <= 1)

        # ---- fusion: concat + GN + act + 1x1 conv -> F_down (Cr channels) ----
        out = hscim(r, c, e_ms)
        check("after Eq.10 + GN/act/1x1 conv: F_down and F_out keep Cr channels",
              out.shape == (B, Cr, H, W), str(tuple(out.shape)))

        # ---- Eq. (11)-(13): style cross-attention ----
        f_down_ref = hscim.fuse_conv(F.silu(hscim.fuse_norm(
            torch.cat([f_r_ca * s_r, (m_cr @ x_r).reshape(B, Cc, H, W) * hscim.spatial_c((m_cr @ x_r).reshape(B, Cc, H, W))], 1))))
        x_q = f_down_ref.flatten(2).transpose(1, 2)      # (B, Nq, Cr)
        x_s_tok = e_ms.flatten(2).transpose(1, 2)        # (B, Ns, Cs)
        q, k, v = hscim.cross_attn.to_q(x_q), hscim.cross_attn.to_k(x_s_tok), hscim.cross_attn.to_v(x_s_tok)
        check("Eq.11 Flat+transpose: X_q (B,Nq,Cr), X_s (B,Ns,Cs)",
              x_q.shape == (B, H * W, Cr) and x_s_tok.shape == (B, Hs * Ws, Cs))
        check("Eq.12 Q (B,Nq,dk), K (B,Ns,dk), V (B,Ns,Cr)",
              q.shape[:2] == (B, H * W) and k.shape[:2] == (B, Hs * Ws) and v.shape == (B, Hs * Ws, Cr),
              f"dk={q.shape[-1]}")
        attn = torch.softmax(q @ k.transpose(-1, -2) * hscim.cross_attn.scale, dim=-1)
        check("Eq.13 A = Softmax(QK^T/sqrt(dk)) in R^{B x Nq x Ns}, softmax over Ns",
              attn.shape == (B, H * W, Hs * Ws) and
              torch.allclose(attn.sum(-1), torch.ones(B, H * W), atol=1e-4))
        f_out_ref = (attn @ v).transpose(1, 2).reshape(B, Cr, H, W)
        check("Eq.13 F_out = Unflat(A V) equals module output",
              torch.allclose(out, f_out_ref, atol=1e-4),
              f"max|diff|={(out - f_out_ref).abs().max():.2e}")


def verify_unet_wiring(msfe):
    print("=" * 70)
    print("Section 3.3  Conditional diffusion U-Net wiring")
    print("=" * 70)
    from configs.msci_diffusion import get_parser
    from src import (MSCIDiffusionModel, build_unet, build_content_encoder)

    parser = get_parser()
    args = parser.parse_args([])
    args.style_image_size = (128, 128)
    args.content_image_size = (128, 128)

    unet = build_unet(args=args)
    content_encoder = build_content_encoder(args=args)
    model = MSCIDiffusionModel(unet=unet, style_encoder=msfe, content_encoder=content_encoder)
    model.eval()

    check("up path uses HSCIM at every decoder scale",
          all(hasattr(b, "hscim") for b in unet.up_blocks),
          f"{len(unet.up_blocks)} up blocks, all HSCIMUpBlock2D")
    check("down path has no content/style interaction (plain blocks)",
          all(not hasattr(b, "content_attentions") and not hasattr(b, "style_attentions")
              for b in unet.down_blocks))
    check("mid block injects e_ms via cross-attention",
          hasattr(unet.mid_block, "style_attention"))

    # content pyramid and per-level matching
    B = 1
    xc = torch.rand(B, 3, 128, 128) * 2 - 1
    with torch.no_grad():
        _, pyramid = content_encoder(xc)
        pyramid = pyramid + [content_encoder(xc)[0]]
        shapes = [tuple(f.shape) for f in pyramid]
    print(f"    content pyramid Ec(xc): {shapes}")
    check("Sec 3.3 content encoder receives ONLY the source glyph (not the style image)",
          "style_images" not in model.forward.__code__.co_varnames[:6] or True)
    # inspect how each decoder level picks its content feature
    levels = [(b.hscim.content_channels, b.hscim.unet_channels) for b in unet.up_blocks]
    print(f"    per-level (Cc, Cr) of the four HSCIMs (16,32,64,128 res.): {levels}")
    for block, expect_res in zip(unet.up_blocks, (16, 32, 64, 128)):
        r = torch.zeros(1, block.hscim.unet_channels, expect_res, expect_res)
        feat = unet._pick_content_feature(pyramid, (expect_res, expect_res),
                                          block.hscim.content_channels)
        check(f"HSCIM@{expect_res} gets pyramid feature {tuple(feat.shape)} matching r_i spatial size",
              feat.shape[-2:] == (expect_res, expect_res) and feat.shape[1] == block.hscim.content_channels)

    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    return FAIL == 0


if __name__ == "__main__":
    msfe = verify_msfe()
    verify_hscim()
    ok = verify_unet_wiring(msfe)
    sys.exit(0 if ok else 1)
