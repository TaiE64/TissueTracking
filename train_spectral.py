"""Phase 2.2/2.3: train SpectralAdapter on SurgT (self-supervised).

Loss = L_photo (warp I_2 by flow → predict I_1, abs diff)
     + 0.1 * L_smooth (edge-aware flow smoothness)
     + 0.01 * L_distill (anchor to RGB-only RAFT prediction)

Trainable: adapter + γ only. MFTIQ + MST++ frozen.

Checkpoints saved every 1000 steps; logs every 50 steps.
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

import sys
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code")
os.chdir(r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")

import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from MFTIQ.config import load_config
from architecture import model_generator as mst_model_generator
from spectral_adapter import FeatureLevelSpectralAdapter, CrossAttentionSpectralAdapter, GatedFusion
from surgt_dataset import SurgTPairDataset

MST_WEIGHTS = r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code/model_zoo/mst_plus_plus.pth"
MFTIQ_CFG = "configs/MFTIQ4_RAFT_200k_cfg.py"
CKPT_DIR = Path(r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/ckpts")


def make_ms_callable(mst):
    def _fn(rgb01):
        _, _, h, w = rgb01.shape
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x_pad = F.pad(rgb01, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = mst(x_pad)[:, :, :h, :w].clamp(0, 1)
        return y
    return _fn


def warp_with_flow(img, flow):
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device),
                            torch.arange(W, device=img.device), indexing="ij")
    grid = torch.stack([xx, yy], dim=0).float().unsqueeze(0).expand(B, -1, -1, -1)
    sample = grid + flow
    sample = sample.permute(0, 2, 3, 1)
    sample[..., 0] = 2 * sample[..., 0] / max(W - 1, 1) - 1
    sample[..., 1] = 2 * sample[..., 1] / max(H - 1, 1) - 1
    return F.grid_sample(img, sample, mode="bilinear", padding_mode="border", align_corners=True)


def edge_aware_smoothness(flow, image):
    dx_f = (flow[..., :, 1:] - flow[..., :, :-1]).abs()
    dy_f = (flow[..., 1:, :] - flow[..., :-1, :]).abs()
    dx_i = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=1, keepdim=True) / 255.0
    dy_i = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=1, keepdim=True) / 255.0
    return (dx_f * torch.exp(-dx_i)).mean() + (dy_f * torch.exp(-dy_i)).mean()


def build_models(adapter_type="self_attn"):
    mst = mst_model_generator("mst_plus_plus", MST_WEIGHTS).cuda().eval()
    for p in mst.parameters():
        p.requires_grad_(False)
    ms_fn = make_ms_callable(mst)

    cfg = load_config(MFTIQ_CFG)
    tracker = cfg.tracker_class(cfg)
    raft = tracker.flower.model
    orig_fnet = raft.fnet
    for p in raft.parameters():
        p.requires_grad_(False)

    if adapter_type == "self_attn":
        adapter = FeatureLevelSpectralAdapter(n_bands=31, trunk_dim=32, n_layers=3,
                                              feat_dim=256, stride=8).cuda()
    elif adapter_type == "cross_attn":
        adapter = CrossAttentionSpectralAdapter(n_bands=31, dim=64, n_heads=4,
                                                feat_dim=256, stride=8).cuda()
    else:
        raise ValueError(f"unknown adapter_type: {adapter_type}")
    gated = GatedFusion(orig_fnet, adapter, gamma_init=1.0,
                        ms_from_rgb=ms_fn, rgb_input_normalized=True).cuda()
    # tiny non-zero init on the final 1x1 conv so all params receive gradient from step 1
    with torch.no_grad():
        if adapter_type == "self_attn":
            nn.init.normal_(adapter.proj.weight, std=1e-4)
            nn.init.normal_(adapter.proj.bias, std=1e-4)
        elif adapter_type == "cross_attn":
            nn.init.normal_(adapter.out_proj.weight, std=1e-4)
            nn.init.normal_(adapter.out_proj.bias, std=1e-4)
    raft.fnet = gated

    # bypass flag for teacher
    gated._bypass = False
    orig_forward = gated.forward
    def fwd_with_bypass(rgb, ms=None):
        if gated._bypass:
            return gated.base(rgb)
        return orig_forward(rgb, ms)
    gated.forward = fwd_with_bypass

    gated.train()
    gated.base.eval()
    for p in gated.base.parameters():
        p.requires_grad_(False)
    for p in gated.adapter.parameters():
        p.requires_grad_(True)
    gated.gamma.requires_grad_(True)

    return tracker, raft, gated, mst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w_photo", type=float, default=1.0)
    parser.add_argument("--w_smooth", type=float, default=0.1)
    parser.add_argument("--w_distill", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="frame spatial scale (0.5=half, 1.0=full res 1280x1024)")
    parser.add_argument("--adapter_type", type=str, default="self_attn",
                        choices=["self_attn", "cross_attn"],
                        help="adapter architecture: self_attn (SpectralFormer-style) or cross_attn (RGB queries MS)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CKPT_DIR / f"train_{args.tag}.log.jsonl"
    summary_path = CKPT_DIR / f"train_{args.tag}.summary.json"

    print(f"[setup] building models ({args.adapter_type}) ...")
    tracker, raft, gated, mst = build_models(adapter_type=args.adapter_type)
    n_train = sum(p.numel() for p in gated.adapter.parameters()) + 1
    print(f"[setup] trainable params: {n_train / 1e6:.3f} M (adapter + γ)")

    opt = torch.optim.AdamW(
        list(gated.adapter.parameters()) + [gated.gamma],
        lr=args.lr, weight_decay=1e-5,
    )

    ds = SurgTPairDataset(side="both", scale=args.scale)  # use BOTH stereo views
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    it = iter(loader)
    print(f"[setup] dataset: {len(ds)} frame pairs from {len(ds.clips)} clips")

    # running averages
    ema = {k: 0.0 for k in ("photo", "smooth", "distill", "total", "gamma", "step_time")}
    EMA_ALPHA = 0.95

    t_start = time.time()
    step = 0
    with open(log_path, "w") as flog:
        while step < args.steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            t0 = time.time()
            img1 = batch["img1"].cuda(non_blocking=True)
            img2 = batch["img2"].cuda(non_blocking=True)

            # student forward (adapter active)
            gated._bypass = False
            out_s = raft(img1, img2, iters=12, test_mode=True)
            flow_s = out_s["flow"]

            # teacher forward (bypass)
            gated._bypass = True
            with torch.no_grad():
                out_t = raft(img1, img2, iters=12, test_mode=True)
            flow_t = out_t["flow"]
            gated._bypass = False

            # losses
            img1_pred = warp_with_flow(img2, flow_s)
            L_photo = (img1_pred - img1).abs().mean() / 255.0
            L_smooth = edge_aware_smoothness(flow_s, img1)
            L_distill = (flow_s - flow_t).pow(2).mean()
            L = args.w_photo * L_photo + args.w_smooth * L_smooth + args.w_distill * L_distill

            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gated.adapter.parameters()) + [gated.gamma], max_norm=1.0)
            opt.step()

            torch.cuda.synchronize()
            step_time = time.time() - t0

            # EMA stats
            vals = {"photo": L_photo.item(), "smooth": L_smooth.item(),
                    "distill": L_distill.item(), "total": L.item(),
                    "gamma": gated.gamma.item(), "step_time": step_time}
            for k, v in vals.items():
                ema[k] = EMA_ALPHA * ema[k] + (1 - EMA_ALPHA) * v if step > 0 else v

            # log
            log_record = {"step": step, **vals,
                          "elapsed_min": (time.time() - t_start) / 60}
            flog.write(json.dumps(log_record) + "\n")
            flog.flush()

            if step % args.log_every == 0 or step == args.steps - 1:
                eta_min = (args.steps - step) * ema["step_time"] / 60
                print(f"[{step:5d}/{args.steps}]  "
                      f"photo={ema['photo']:.4f}  smooth={ema['smooth']:.4f}  "
                      f"distill={ema['distill']:.5f}  total={ema['total']:.4f}  "
                      f"γ={ema['gamma']:.4f}  "
                      f"{ema['step_time']:.2f}s/step  ETA {eta_min:.0f}min")

            # ckpt
            if (step + 1) % args.ckpt_every == 0 or step == args.steps - 1:
                ckpt = {
                    "step": step + 1,
                    "adapter": gated.adapter.state_dict(),
                    "gamma": gated.gamma.detach().cpu(),
                    "args": vars(args),
                }
                torch.save(ckpt, CKPT_DIR / f"{args.tag}_step{step + 1:06d}.pt")
                print(f"  [ckpt] saved {args.tag}_step{step + 1:06d}.pt")

            step += 1

    summary = {
        "args": vars(args),
        "total_min": (time.time() - t_start) / 60,
        "final_ema": ema,
        "ckpt_dir": str(CKPT_DIR),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] total {summary['total_min']:.1f} min. Summary at {summary_path}")


if __name__ == "__main__":
    main()
