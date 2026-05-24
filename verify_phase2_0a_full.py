"""Phase 2.0a (FULL) — wire real MST++ MS into GatedFusion via on-the-fly
derivation, and verify spectral signal genuinely propagates to MFTIQ output.

Three passes on the same STIR seq05 clip:
  Pass 1: pristine RGB-only MFTIQ              → endpoint A
  Pass 2: GatedFusion + zero-init + REAL MST++ → endpoint B  (must == A bit-exactly)
  Pass 3: GatedFusion + PERTURBED adapter + REAL MST++ → endpoint C (must != A)

Pass 2 vs Pass 1 confirms the zero-init pipeline is bit-clean.
Pass 3 vs Pass 1 confirms spectral signal actually affects tracking output.
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
from tempfile import NamedTemporaryFile

from MFTIQ.config import load_config
from MFTIQ.point_tracking import convert_to_point_tracking
import MFTIQ.utils.io as io_utils

from architecture import model_generator as mst_model_generator
from spectral_adapter import FeatureLevelSpectralAdapter, GatedFusion


SEQ_DIR = r"c:/Users/29421/Desktop/TissueTracking/STIRChallenge_2024/02/left/seq05"
VIDEO_PATH = f"{SEQ_DIR}/frames/00078760ms-00081600ms-visible.mp4"
SEG_DIR = f"{SEQ_DIR}/segmentation"
MST_WEIGHTS = r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code/model_zoo/mst_plus_plus.pth"
MFTIQ_CFG = "configs/MFTIQ4_RAFT_200k_cfg.py"


def extract_centroids(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def downsample_video(video_path, scale=0.5):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_W = int(W * scale) & ~1
    new_H = int(H * scale) & ~1
    t = NamedTemporaryFile(delete=False, suffix=".mp4")
    t.close()
    writer = cv2.VideoWriter(t.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_W, new_H))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(f, (new_W, new_H), interpolation=cv2.INTER_AREA))
    cap.release()
    writer.release()
    return t.name, (new_W / W, new_H / H)


def track_one(video_path, query_xy, tracker):
    queries = torch.from_numpy(query_xy).float().cuda()
    last_coords = None
    initialized = False
    for frame in io_utils.get_video_frames(str(video_path)):
        if not initialized:
            tracker.init(frame)
            initialized = True
        else:
            meta = tracker.track(frame)
            coords, _ = convert_to_point_tracking(meta.result, queries)
            last_coords = coords
    if last_coords is None:
        return None
    return last_coords.cpu().numpy() if hasattr(last_coords, "cpu") else np.asarray(last_coords)


def make_ms_callable(mst_model):
    """Return a callable: rgb01 in (B,3,H,W) -> ms in (B,31,H,W)."""
    def _fn(rgb01):
        x = rgb01
        _, _, h, w = x.shape
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x_pad = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = mst_model(x_pad)[:, :, :h, :w].clamp(0, 1)
        return y
    return _fn


def main():
    print("== Preparing inputs ==")
    small_video, (sx, sy) = downsample_video(VIDEO_PATH, scale=0.5)
    start = extract_centroids(Path(SEG_DIR) / "icgstartseg.png")
    start_small = start.copy()
    start_small[:, 0] *= sx
    start_small[:, 1] *= sy
    print(f"  start centroid (small res): {start_small.tolist()}")

    mst = mst_model_generator("mst_plus_plus", MST_WEIGHTS).cuda().eval()
    for p in mst.parameters():
        p.requires_grad_(False)
    ms_fn = make_ms_callable(mst)

    # ─────────────────────────────────────────────────────────────────────
    # Pass 1: pristine RGB-only
    # ─────────────────────────────────────────────────────────────────────
    print("\n== Pass 1: pristine RGB-only ==")
    cfg = load_config(MFTIQ_CFG)
    tracker_a = cfg.tracker_class(cfg)
    pred_a = track_one(small_video, start_small, tracker_a)
    print(f"  endpoint A: {pred_a.tolist()}")
    del tracker_a
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────
    # Pass 2: GatedFusion + zero-init + real MS via MST++
    # ─────────────────────────────────────────────────────────────────────
    print("\n== Pass 2: GatedFusion zero-init + REAL MST++ MS ==")
    cfg = load_config(MFTIQ_CFG)
    tracker_b = cfg.tracker_class(cfg)
    raft_b = tracker_b.flower.model
    original_fnet_b = raft_b.fnet
    adapter_b = FeatureLevelSpectralAdapter(n_bands=31, trunk_dim=32, n_layers=3,
                                            feat_dim=256, stride=8).cuda().eval()
    gated_b = GatedFusion(original_fnet_b, adapter_b, gamma_init=1.0,
                          ms_from_rgb=ms_fn, rgb_input_normalized=True).cuda().eval()
    raft_b.fnet = gated_b

    pred_b = track_one(small_video, start_small, tracker_b)
    diff_b = np.abs(pred_a - pred_b).max()
    print(f"  endpoint B: {pred_b.tolist()}")
    print(f"  diff vs A: {diff_b:.6e}  (must be 0.0 — zero-init must not change anything)")
    assert diff_b == 0.0, f"zero-init regression: expected 0, got {diff_b}"
    print(f"  PASS")
    del tracker_b
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────
    # Pass 3: GatedFusion + PERTURBED adapter + real MS
    # ─────────────────────────────────────────────────────────────────────
    print("\n== Pass 3: GatedFusion PERTURBED adapter + REAL MST++ MS ==")
    cfg = load_config(MFTIQ_CFG)
    tracker_c = cfg.tracker_class(cfg)
    raft_c = tracker_c.flower.model
    original_fnet_c = raft_c.fnet
    adapter_c = FeatureLevelSpectralAdapter(n_bands=31, trunk_dim=32, n_layers=3,
                                            feat_dim=256, stride=8).cuda().eval()
    # Perturb head[-1] with small random — breaks zero-init
    with torch.no_grad():
        nn.init.normal_(adapter_c.proj.weight, std=0.05)
        nn.init.normal_(adapter_c.proj.bias, std=0.05)
    gated_c = GatedFusion(original_fnet_c, adapter_c, gamma_init=1.0,
                          ms_from_rgb=ms_fn, rgb_input_normalized=True).cuda().eval()
    raft_c.fnet = gated_c

    pred_c = track_one(small_video, start_small, tracker_c)
    diff_c = np.abs(pred_a - pred_c).max()
    print(f"  endpoint C: {pred_c.tolist()}")
    print(f"  diff vs A: {diff_c:.4e}  (must be > 0 — spectral signal should perturb tracking)")
    if diff_c > 1e-3:
        print(f"  PASS: spectral signal genuinely affects MFTIQ predictions")
    else:
        print(f"  FAIL: tracker output unchanged — spectral signal may be ignored downstream")

    try:
        os.unlink(small_video)
    except Exception:
        pass


if __name__ == "__main__":
    main()
