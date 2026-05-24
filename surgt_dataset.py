"""SurgT_train frame-pair dataset for self-supervised training.

Each item is an adjacent-frame pair (I_t, I_{t+1}) from one stereo SurgT clip.
SurgT videos are vertically stacked stereo: top half is the LEFT camera,
bottom half is the RIGHT camera. By default we use only the LEFT view.

Output format per __getitem__:
  {
    "img1": Tensor (3, H, W) float, values in [0, 255]   # matches RAFT input convention
    "img2": Tensor (3, H, W) float, values in [0, 255]
    "clip": str (path), "frame_idx": int
  }
"""
import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_clips(root: Path, required_resolution=(1280, 2048)) -> List[Tuple[Path, int]]:
    """Return list of (video_path, n_frames). Skips unreadable clips or clips
    whose (W, H) doesn't match required_resolution (SurgT has a mix of resolutions;
    only the (1280, 2048) clips are standard vertical-stacked stereo at full res).
    """
    out = []
    skipped = 0
    for video in sorted(root.rglob("video.mp4")):
        cap = cv2.VideoCapture(str(video))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if (W, H) != required_resolution or n < 2:
            skipped += 1
            continue
        out.append((video, n))
    if skipped:
        print(f"[SurgTPairDataset] skipped {skipped} clips not at {required_resolution}")
    return out


class SurgTPairDataset(Dataset):
    def __init__(
        self,
        root: str = r"c:/Users/29421/Desktop/TissueTracking/SurgT_train/train",
        side: str = "left",          # "left", "right", or "both"
        scale: float = 0.5,           # spatial downsample factor (output ≈ 640x512)
        size_multiple: int = 8,       # round dims down to multiple of this (RAFT needs /8)
        max_pairs_per_clip: Optional[int] = None,
    ):
        super().__init__()
        self.root = Path(root)
        assert self.root.is_dir(), f"root not found: {self.root}"
        assert side in ("left", "right", "both")
        self.side = side
        self.scale = scale
        self.size_multiple = size_multiple

        clips = _list_clips(self.root)
        assert clips, f"no video.mp4 found under {self.root}"

        # build flat index: list of (clip_idx, frame_idx)
        # for each clip, valid frame_idx is [0, n_frames-2]
        self.index: List[Tuple[int, int]] = []
        for ci, (_, n) in enumerate(clips):
            pairs = n - 1
            if max_pairs_per_clip is not None:
                pairs = min(pairs, max_pairs_per_clip)
            for fi in range(pairs):
                self.index.append((ci, fi))

        self.clips = clips
        # if side="both", duplicate index for the other side
        if side == "both":
            self._n_left = len(self.index)
            self.index = self.index + self.index  # second half is "right"

    def __len__(self):
        return len(self.index)

    def _crop_side(self, frame_bgr, idx):
        """frame is vertical stereo stack. Choose top half (left) or bottom (right)."""
        H = frame_bgr.shape[0]
        half = H // 2
        if self.side == "left":
            return frame_bgr[:half]
        elif self.side == "right":
            return frame_bgr[half:]
        else:  # both
            return frame_bgr[:half] if idx < self._n_left else frame_bgr[half:]

    def _resize_to_target(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        new_w = int(w * self.scale) & ~(self.size_multiple - 1)
        new_h = int(h * self.scale) & ~(self.size_multiple - 1)
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def __getitem__(self, idx):
        ci, fi = self.index[idx]
        video_path, _ = self.clips[ci]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok1, bgr1 = cap.read()
        ok2, bgr2 = cap.read()
        cap.release()
        if not (ok1 and ok2):
            # fall back to a different index — recursive once
            return self.__getitem__((idx + 1) % len(self))

        bgr1 = self._crop_side(bgr1, idx)
        bgr2 = self._crop_side(bgr2, idx)
        bgr1 = self._resize_to_target(bgr1)
        bgr2 = self._resize_to_target(bgr2)

        rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(rgb1).permute(2, 0, 1).float()  # (3, H, W), [0, 255]
        t2 = torch.from_numpy(rgb2).permute(2, 0, 1).float()
        return {
            "img1": t1,
            "img2": t2,
            "clip": str(video_path),
            "frame_idx": fi,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = SurgTPairDataset(side="left", scale=0.5)
    print(f"clips: {len(ds.clips)}")
    print(f"total frame pairs: {len(ds)}")
    # per-clip pair count summary
    counts = {}
    for ci, _ in ds.index:
        counts.setdefault(ci, 0)
        counts[ci] += 1
    print(f"  pairs/clip: min={min(counts.values())}, max={max(counts.values())}, "
          f"median={sorted(counts.values())[len(counts) // 2]}")

    # sanity samples
    rng = random.Random(0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for row in range(3):
        idx = rng.randrange(len(ds))
        s = ds[idx]
        axes[row, 0].imshow(s["img1"].permute(1, 2, 0).numpy().astype(np.uint8))
        axes[row, 0].set_title(f"idx={idx}  clip={Path(s['clip']).parent.parent.name}/{Path(s['clip']).parent.name}  frame {s['frame_idx']}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(s["img2"].permute(1, 2, 0).numpy().astype(np.uint8))
        axes[row, 1].set_title(f"frame {s['frame_idx'] + 1}")
        axes[row, 1].axis("off")
        print(f"  sample idx={idx}: shape {tuple(s['img1'].shape)}, range [{s['img1'].min():.0f}, {s['img1'].max():.0f}]")
    plt.tight_layout()
    out = r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/surgt_dataset_sanity.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
