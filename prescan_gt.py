"""Pre-scan all 120 STIR seqs: count GT points (start/end centroids) per seq.
Output a JSON so the full eval can skip seqs with 0 points."""
import json
from pathlib import Path
import cv2
import numpy as np

STIR_ROOT = r"c:/Users/29421/Desktop/TissueTracking/STIRChallenge_2024"
OUT = r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/gt_scan.json"


def count_centroids(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4)


result = {}
total_pts_start = 0
total_pts_end = 0
n_valid = 0
n_total = 0
for col_dir in sorted(Path(STIR_ROOT).iterdir()):
    if not col_dir.is_dir():
        continue
    for side in ["left", "right"]:
        side_dir = col_dir / side
        if not side_dir.is_dir():
            continue
        for seq_dir in sorted(side_dir.glob("seq*")):
            n_total += 1
            seg_dir = seq_dir / "segmentation"
            n_start = count_centroids(seg_dir / "icgstartseg.png")
            n_end = count_centroids(seg_dir / "icgendseg.png")
            key = f"{col_dir.name}_{side}_{seq_dir.name}"
            result[key] = {"n_start": n_start, "n_end": n_end}
            if n_start > 0 and n_end > 0:
                n_valid += 1
                total_pts_start += n_start
                total_pts_end += n_end

print(f"Total seqs: {n_total}")
print(f"Valid seqs (both start>0 and end>0): {n_valid}")
print(f"Total start centroids: {total_pts_start}")
print(f"Total end centroids:   {total_pts_end}")

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"saved {OUT}")
