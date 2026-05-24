"""STIR eval with downscaled frames to fit MFTIQ in reasonable time.

Pipeline per seq:
  - Read original video, downscale frames to half resolution
  - Scale query centroids accordingly
  - Run MFTIQ at low res
  - Scale predicted positions back to original res
  - Compare against original-res GT centroids

Scope: 30 seqs sampled across all 9 collections (3 per collection).
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

import sys
import json
import time
import traceback
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from tempfile import NamedTemporaryFile

MFTIQ_ROOT = r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master"
STIR_ROOT = r"c:/Users/29421/Desktop/TissueTracking/STIRChallenge_2024"
FALSE_VID_DIR = r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/false_videos"
OUT_DIR = r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/resized_eval"

sys.path.insert(0, MFTIQ_ROOT)
os.chdir(MFTIQ_ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

from MFTIQ.config import load_config as mft_load_config
from MFTIQ.point_tracking import convert_to_point_tracking
import MFTIQ.utils.io as io_utils

MFTIQ_CFG = "configs/MFTIQ4_RAFT_200k_cfg.py"
THRESHOLDS_PX = [4, 8, 16, 32, 64]
SCALE = 0.5  # downscale frames by 2x (1280x1024 -> 640x512)
SEQS_PER_COLLECTION = 3


def extract_centroids(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((0, 2), dtype=np.float32)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def write_resized_mp4(input_video, scale):
    """Write a half-resolution mp4 to a temp file. Returns path."""
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_W = int(W * scale) & ~1  # even
    new_H = int(H * scale) & ~1
    tmp = NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_W, new_H))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_AREA)
        writer.write(small)
    cap.release()
    writer.release()
    return tmp.name, (new_W / W, new_H / H)


def track_one(video_path, query_xy, cfg):
    tracker = cfg.tracker_class(cfg)
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
    del tracker
    torch.cuda.empty_cache()
    if last_coords is None:
        return None
    return last_coords.cpu().numpy() if hasattr(last_coords, "cpu") else np.asarray(last_coords)


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
    """Sample seqs evenly across collections + both sides."""
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
        # sample per_collection
        if len(col_seqs) > per_collection:
            idx = rng.choice(len(col_seqs), per_collection, replace=False)
            col_seqs = [col_seqs[i] for i in sorted(idx)]
        seqs.extend(col_seqs)
    return seqs


def main():
    seqs = discover_seqs()
    print(f"Using {len(seqs)} sampled seqs ({SEQS_PER_COLLECTION} per collection)")
    print(f"Frame scale factor: {SCALE} (1280x1024 -> {int(1280*SCALE)}x{int(1024*SCALE)})")

    cfg = mft_load_config(MFTIQ_CFG)
    all_dists = {"orig": [], "natural": [], "maxspread": []}
    per_seq = {}
    n_evaled = 0
    n_failed = 0
    t0 = time.time()

    for s in tqdm(seqs, desc="seqs"):
        try:
            start = extract_centroids(s["seg_dir"] / "icgstartseg.png")
            end = extract_centroids(s["seg_dir"] / "icgendseg.png")
            if len(start) == 0 or len(end) == 0:
                continue
            per_seq[s["key"]] = {"n_start": int(len(start)), "n_end": int(len(end)),
                                 "orig_dists": [], "natural_dists": [], "maxspread_dists": []}

            for variant in ["orig", "natural", "maxspread"]:
                if variant == "orig":
                    src = s["video"]
                else:
                    src = Path(FALSE_VID_DIR) / f"{s['key']}_{variant}.mp4"
                if not Path(src).exists():
                    continue

                # Write resized video to temp
                small_path, (sx, sy) = write_resized_mp4(src, SCALE)
                try:
                    # Scale start centroids to low res
                    start_small = start.copy()
                    start_small[:, 0] *= sx
                    start_small[:, 1] *= sy
                    pred_small = track_one(small_path, start_small, cfg)
                    if pred_small is None:
                        continue
                    # Scale predicted back to original res
                    pred = pred_small.copy()
                    pred[:, 0] /= sx
                    pred[:, 1] /= sy
                    d = match_dists(pred, end)
                    all_dists[variant].extend(d.tolist())
                    per_seq[s["key"]][f"{variant}_dists"] = d.tolist()
                finally:
                    try:
                        os.unlink(small_path)
                    except Exception:
                        pass
            n_evaled += 1
            # save partial every 5 seqs
            if n_evaled % 5 == 0:
                with open(f"{OUT_DIR}/partial.json", "w") as f:
                    json.dump({"per_variant_dists": all_dists, "per_seq": per_seq,
                              "n_evaled": n_evaled, "elapsed_min": (time.time() - t0) / 60}, f, indent=2)
                print(f"  partial saved at {n_evaled} seqs, elapsed {(time.time() - t0)/60:.1f}min")
        except Exception as e:
            n_failed += 1
            print(f"\nfailed on {s['key']}: {e}")
            traceback.print_exc()

    # Aggregate
    print("\n== aggregate ==")
    results = {"thresholds_px": THRESHOLDS_PX, "per_variant_acc": {}, "per_variant_dists": {}, "per_seq": per_seq,
               "scale": SCALE, "n_seqs": n_evaled}
    for variant, dists in all_dists.items():
        arr = np.array(dists)
        acc = {t: float((arr < t).mean()) if len(arr) else float("nan") for t in THRESHOLDS_PX}
        avg_acc = float(np.nanmean([acc[t] for t in THRESHOLDS_PX])) if len(arr) else float("nan")
        results["per_variant_acc"][variant] = acc
        results["per_variant_dists"][variant] = {
            "n_points": len(arr),
            "median_px": float(np.median(arr)) if len(arr) else float("nan"),
            "mean_px": float(np.mean(arr)) if len(arr) else float("nan"),
            "avg_acc": avg_acc,
        }
        print(f"  {variant:10s}: n={len(arr):4d}  median={np.median(arr) if len(arr) else 0:6.1f}px  "
              f"acc@4/8/16/32/64 = {[f'{acc[t]:.3f}' for t in THRESHOLDS_PX]}  avg={avg_acc:.3f}")

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"orig": "C0", "natural": "C1", "maxspread": "C2"}

    ax = axes[0]
    for variant in ["orig", "natural", "maxspread"]:
        accs = [results["per_variant_acc"][variant][t] for t in THRESHOLDS_PX]
        ax.plot(THRESHOLDS_PX, accs, "o-",
                label=f"{variant} (avg={results['per_variant_dists'][variant]['avg_acc']:.3f})",
                linewidth=2, markersize=8, color=colors[variant])
    ax.set_xlabel("pixel threshold (px)")
    ax.set_ylabel("accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xticks(THRESHOLDS_PX)
    ax.set_xticklabels(THRESHOLDS_PX)
    n_pts = results["per_variant_dists"]["orig"]["n_points"]
    ax.set_title(f"STIR endpoint accuracy (n={n_pts} pts across {n_evaled} seqs, frame scale {SCALE})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for variant in ["orig", "natural", "maxspread"]:
        d = np.array(all_dists[variant])
        if len(d) == 0:
            continue
        ax.hist(np.clip(d, 0, 200), bins=40, alpha=0.5,
                label=f"{variant} (med={np.median(d):.1f})", color=colors[variant])
    ax.set_xlabel("endpoint error (px, clipped at 200)")
    ax.set_ylabel("count")
    ax.set_title("error distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {OUT_DIR}/results.json and {OUT_DIR}/comparison.png")
    print(f"Total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
