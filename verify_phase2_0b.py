"""Phase 2.0b — verify gradient flow through RAFT.forward to SpectralAdapter.

Critical decision point: can endpoint/flow loss backprop to adapter params?
We're now using the SIMPLER training path: single RAFT.forward on a frame pair
(no MFTIQ.track multi-delta selection, no argmax). This is fully differentiable.

Test plan:
  1. Load real MFTIQ tracker → get RAFT model
  2. Wrap fnet with GatedFusion(adapter) + ms_from_rgb = MST++
  3. Sample two consecutive frames from a SurgT clip
  4. Run RAFT.forward, get flow prediction
  5. Compute a dummy loss (sum of flow squared)
  6. backward()
  7. Check: adapter.proj.weight.grad is not None AND nonzero
            adapter.trunk has gradient
            γ has gradient
            base fnet has no gradient (frozen)
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

import sys
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code")
os.chdir(r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from MFTIQ.config import load_config
from architecture import model_generator as mst_model_generator
from spectral_adapter import FeatureLevelSpectralAdapter, GatedFusion


SURGT_CLIP = r"c:/Users/29421/Desktop/TissueTracking/SurgT_train/train/case_1/1/video.mp4"
MST_WEIGHTS = r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code/model_zoo/mst_plus_plus.pth"
MFTIQ_CFG = "configs/MFTIQ4_RAFT_200k_cfg.py"


def load_two_consecutive_frames_half_res(video_path, frame_idx=0):
    """Load frames frame_idx and frame_idx+1, downsample to half res, return as (1,3,H,W) tensors in [0, 255]."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok1, bgr1 = cap.read()
    ok2, bgr2 = cap.read()
    cap.release()
    assert ok1 and ok2, "failed to read two consecutive frames"
    # SurgT is stereo vertical-stacked; take TOP half as 'left' camera
    H_full = bgr1.shape[0]
    bgr1 = bgr1[:H_full // 2]
    bgr2 = bgr2[:H_full // 2]
    # downsample to half
    h, w = bgr1.shape[:2]
    new_w, new_h = (w // 2) & ~7, (h // 2) & ~7  # multiple of 8 for RAFT
    bgr1 = cv2.resize(bgr1, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr2 = cv2.resize(bgr2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).float().cuda()  # in [0, 255]
    t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).float().cuda()
    return t1, t2


def make_ms_callable(mst_model):
    def _fn(rgb01):
        _, _, h, w = rgb01.shape
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x_pad = torch.nn.functional.pad(rgb01, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = mst_model(x_pad)[:, :, :h, :w].clamp(0, 1)
        return y
    return _fn


def main():
    print("== Loading frames + MST++ + MFTIQ tracker ==")
    img1, img2 = load_two_consecutive_frames_half_res(SURGT_CLIP, frame_idx=10)
    print(f"  frame shape: {img1.shape}, value range: [{img1.min().item():.1f}, {img1.max().item():.1f}]")

    mst = mst_model_generator("mst_plus_plus", MST_WEIGHTS).cuda().eval()
    for p in mst.parameters():
        p.requires_grad_(False)
    ms_fn = make_ms_callable(mst)

    cfg = load_config(MFTIQ_CFG)
    tracker = cfg.tracker_class(cfg)
    raft_model = tracker.flower.model
    original_fnet = raft_model.fnet
    # freeze RAFT base
    for p in raft_model.parameters():
        p.requires_grad_(False)

    adapter = FeatureLevelSpectralAdapter(n_bands=31, trunk_dim=32, n_layers=3,
                                          feat_dim=256, stride=8).cuda()
    gated = GatedFusion(original_fnet, adapter, gamma_init=1.0,
                        ms_from_rgb=ms_fn, rgb_input_normalized=True).cuda()
    raft_model.fnet = gated

    # Enable training mode for adapter, keep base frozen
    gated.train()
    gated.base.eval()  # keep BN/IN in eval mode
    for p in gated.base.parameters():
        p.requires_grad_(False)
    for p in gated.adapter.parameters():
        p.requires_grad_(True)
    gated.gamma.requires_grad_(True)

    print("\n== Running RAFT.forward + backward ==")
    # Perturb head[-1] slightly so ΔF is non-zero (so γ also gets gradient on first pass)
    with torch.no_grad():
        nn.init.normal_(adapter.proj.weight, std=0.001)
        nn.init.normal_(adapter.proj.bias, std=0.001)

    # RAFT expects float tensors. test_mode=True returns full flow predictions.
    # Run with grad enabled (default is no_grad in the wrapper but model itself is fine)
    out = raft_model(img1, img2, iters=12, test_mode=True)
    # out is a dict with 'flow' (1, 2, H, W), 'occlusion', 'uncertainty'
    flow = out["flow"]
    print(f"  flow shape: {flow.shape}, requires_grad: {flow.requires_grad}")
    assert flow.requires_grad, "RAFT output must require grad for adapter training"

    # Dummy loss
    loss = flow.pow(2).mean()
    print(f"  loss: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    head_last_w_grad = adapter.proj.weight.grad
    head_last_b_grad = adapter.proj.bias.grad
    trunk_grad_sum = sum(p.grad.abs().sum().item() if p.grad is not None else 0
                         for p in adapter.trunk.parameters())
    head_other_grad_sum = sum(p.grad.abs().sum().item() if p.grad is not None else 0
                              for p in adapter.spatial_down.parameters())
    gamma_grad = gated.gamma.grad
    base_has_grad = any(p.grad is not None for p in original_fnet.parameters())

    print("\n== Gradient flow check ==")
    print(f"  adapter.proj.weight.grad norm: {head_last_w_grad.abs().sum().item():.4f}  (need > 0)")
    print(f"  adapter.proj.bias.grad norm:   {head_last_b_grad.abs().sum().item():.4f}  (need > 0)")
    print(f"  adapter.trunk grad sum:            {trunk_grad_sum:.4f}  (need > 0 once head moves)")
    print(f"  adapter.spatial_down grad sum:        {head_other_grad_sum:.4f}  (need > 0)")
    print(f"  γ grad: {gamma_grad.item():.4e}  (need != 0)")
    print(f"  base fnet has any grad: {base_has_grad}  (must be False — frozen)")

    head_last_alive = head_last_w_grad.abs().sum().item() > 0
    gamma_alive = gamma_grad.abs().item() > 1e-12
    trunk_alive = trunk_grad_sum > 0
    head_other_alive = head_other_grad_sum > 0

    print("\n== Verdict ==")
    print(f"  head[-1] gets gradient: {head_last_alive}")
    print(f"  γ gets gradient: {gamma_alive}")
    print(f"  trunk gets gradient: {trunk_alive}")
    print(f"  head[:-1] gets gradient: {head_other_alive}")
    print(f"  base remains frozen: {not base_has_grad}")
    if head_last_alive and gamma_alive and trunk_alive and head_other_alive and not base_has_grad:
        print("\n  [PASS] ALL CHECKS PASS - training loop is feasible.")
    else:
        print("\n  [FAIL] Something's broken - gradient flow not as expected.")


if __name__ == "__main__":
    main()
