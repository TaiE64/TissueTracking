"""Zero-shot evaluate a SurgT-trained SpectralAdapter on STIR.

Loads adapter checkpoint, wraps MFTIQ fnet with it, runs the same 25-seq
half-res STIR evaluation pipeline as run_resized_eval.py, and compares.
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code")
os.chdir(r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")

from MFTIQ.config import load_config
from MFTIQ.point_tracking import convert_to_point_tracking
import MFTIQ.utils.io as io_utils

from architecture import model_generator as mst_model_generator
from spectral_adapter import FeatureLevelSpectralAdapter, GatedFusion


STIR_ROOT = r"c:/Users/29421/Desktop/TissueTracking/STIRChallenge_2024"
MST_WEIGHTS = r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code/model_zoo/mst_plus_plus.pth"
MFTIQ_CFG = "configs/MFTIQ4_RAFT_200k_cfg.py"
THRESHOLDS_PX = [4, 8, 16, 32, 64]
SCALE = 0.5  # default, can be overridden by --scale CLI flag
SEQS_PER_COLLECTION = 3  # same sampling seed=42 as baseline


def extract_centroids(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def downsample_video(video_path, scale=SCALE):
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


def match_dists(pred_xy, gt_xy):
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.array([])
    D = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    if D.shape[0] != D.shape[1]:
        big = max(D.shape) + 1
        Dp = np.full((big, big), 1e9, dtype=np.float64)
        Dp[:D.shape[0], :D.shape[1]] = D
        D = Dp
    pi, gi = linear_sum_assignment(D)
    return np.array([np.linalg.norm(pred_xy[p] - gt_xy[g])
                     for p, g in zip(pi, gi) if p < len(pred_xy) and g < len(gt_xy)])


def discover_seqs(per_collection=SEQS_PER_COLLECTION):
    rng = np.random.default_rng(42)
    seqs = []
    for col_dir in sorted(Path(STIR_ROOT).iterdir()):
        if not col_dir.is_dir():
            continue
        col_seqs = []
        for side in ["left", "right"]:
            side_dir = col_dir / side
            if not side_dir.is_dir():
                continue
            for seq_dir in sorted(side_dir.glob("seq*")):
                vids = list((seq_dir / "frames").glob("*.mp4"))
                if not vids:
                    continue
                col_seqs.append({"col": col_dir.name, "side": side, "name": seq_dir.name,
                                 "video": vids[0], "seg_dir": seq_dir / "segmentation",
                                 "key": f"{col_dir.name}_{side}_{seq_dir.name}"})
        if len(col_seqs) > per_collection:
            idx = rng.choice(len(col_seqs), per_collection, replace=False)
            col_seqs = [col_seqs[i] for i in sorted(idx)]
        seqs.extend(col_seqs)
    return seqs


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


def build_tracker_with_adapter(ckpt_path):
    """Build a fresh MFTIQ tracker, wrap fnet with GatedFusion containing the
    trained adapter. Returns the tracker (ready for MFTIQ.track API)."""
    mst = mst_model_generator("mst_plus_plus", MST_WEIGHTS).cuda().eval()
    for p in mst.parameters():
        p.requires_grad_(False)
    ms_fn = make_ms_callable(mst)

    cfg = load_config(MFTIQ_CFG)
    tracker = cfg.tracker_class(cfg)
    raft = tracker.flower.model
    orig_fnet = raft.fnet
    adapter = FeatureLevelSpectralAdapter(n_bands=31, trunk_dim=32, n_layers=3,
                                          feat_dim=256, stride=8).cuda().eval()
    gated = GatedFusion(orig_fnet, adapter, gamma_init=1.0,
                        ms_from_rgb=ms_fn, rgb_input_normalized=True).cuda().eval()

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cuda")
        gated.adapter.load_state_dict(ckpt["adapter"])
        gated.gamma.data.copy_(ckpt["gamma"].to(gated.gamma.device))
        print(f"  loaded ckpt step={ckpt['step']}, γ={gated.gamma.item():.4f}")
    else:
        print(f"  no ckpt — running with random/zero-init adapter")

    raft.fnet = gated
    return tracker, mst  # keep mst alive (ms_fn references it)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to ckpt .pt (or 'none' for untrained adapter baseline)")
    parser.add_argument("--tag", type=str, default="trained",
                        help="output filename prefix")
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="evaluate only first N seqs (0 = all)")
    args = parser.parse_args()

    ckpt_path = None if args.ckpt.lower() == "none" else args.ckpt
    out_dir = Path(r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/zero_shot_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== building tracker with adapter (ckpt={ckpt_path}) ==")
    tracker, mst = build_tracker_with_adapter(ckpt_path)

    seqs = discover_seqs()
    if args.max_seqs > 0:
        seqs = seqs[:args.max_seqs]
    print(f"== evaluating on {len(seqs)} sampled STIR seqs ==")
    all_dists = []
    per_seq = {}
    n_evaled = 0
    t0 = time.time()
    for s in tqdm(seqs):
        try:
            start = extract_centroids(s["seg_dir"] / "icgstartseg.png")
            end = extract_centroids(s["seg_dir"] / "icgendseg.png")
            if len(start) == 0 or len(end) == 0:
                continue
            small_path, (sx, sy) = downsample_video(s["video"], SCALE)
            try:
                start_small = start.copy()
                start_small[:, 0] *= sx
                start_small[:, 1] *= sy
                pred_small = track_one(small_path, start_small, tracker)
                if pred_small is None:
                    continue
                pred = pred_small.copy()
                pred[:, 0] /= sx
                pred[:, 1] /= sy
                d = match_dists(pred, end)
                all_dists.extend(d.tolist())
                per_seq[s["key"]] = {"n_start": int(len(start)), "n_end": int(len(end)),
                                     "distances_px": d.tolist()}
            finally:
                try: os.unlink(small_path)
                except: pass
            n_evaled += 1
            # incremental partial save so we can peek progress
            partial_arr = np.array(all_dists)
            partial = {
                "n_seqs_so_far": n_evaled,
                "per_seq": per_seq,
                "median_so_far": float(np.median(partial_arr)) if len(partial_arr) else float("nan"),
                "acc_so_far": {t: float((partial_arr < t).mean()) if len(partial_arr) else float("nan") for t in THRESHOLDS_PX},
            }
            with open(out_dir / f"{args.tag}_partial.json", "w") as f:
                json.dump(partial, f, indent=2)
            print(f"  [{n_evaled}/{len(seqs)}] {s['key']}: dists={[round(x,1) for x in d.tolist()]}")
        except Exception as e:
            print(f"\nfailed on {s['key']}: {e}")
            traceback.print_exc()

    arr = np.array(all_dists)
    acc = {t: float((arr < t).mean()) if len(arr) else float("nan") for t in THRESHOLDS_PX}
    avg_acc = float(np.nanmean([acc[t] for t in THRESHOLDS_PX])) if len(arr) else float("nan")
    res = {
        "ckpt": str(ckpt_path),
        "n_seqs": n_evaled,
        "n_points": len(arr),
        "median_px": float(np.median(arr)) if len(arr) else float("nan"),
        "mean_px": float(np.mean(arr)) if len(arr) else float("nan"),
        "acc_per_thresh": acc,
        "avg_acc": avg_acc,
        "per_seq": per_seq,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = out_dir / f"{args.tag}_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n== Summary ==")
    print(f"  ckpt: {ckpt_path}")
    print(f"  n_points: {res['n_points']}  median: {res['median_px']:.2f}px  mean: {res['mean_px']:.2f}px")
    print(f"  acc@4/8/16/32/64: {[f'{acc[t]:.3f}' for t in THRESHOLDS_PX]}")
    print(f"  avg acc: {avg_acc:.4f}")
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
