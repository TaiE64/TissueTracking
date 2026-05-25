"""Phase A1: SUPERVISED training of SpectralAdapter on STIR train split.

For each STIR seq with GT (start_xy, end_xy):
  - Decode all frames at half-resolution
  - Chain RAFT-with-adapter through consecutive frame pairs
  - At each pair: sample predicted flow at current point, update position
  - At the last frame: predicted end position
  - Loss = L2 distance between predicted end and GT end

Gradient checkpointing per frame pair to keep memory bounded.
Adapter (self_attn or cross_attn) is the only trainable component.

Train: 80 STIR seqs (random subset of the 95 non-test)
Val:   15 STIR seqs
Test:  the same 25 seqs we have been evaluating on (NEVER seen during training)
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

import sys
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code")
os.chdir(r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")

import json
import time
import argparse
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

from MFTIQ.config import load_config
from architecture import model_generator as mst_model_generator
from spectral_adapter import FeatureLevelSpectralAdapter, CrossAttentionSpectralAdapter, GatedFusion

STIR_ROOT = r"c:/Users/29421/Desktop/TissueTracking/STIRChallenge_2024"
MST_WEIGHTS = r"c:/Users/29421/Desktop/TissueTracking/MST-plus-plus-master/MST-plus-plus-master/predict_code/model_zoo/mst_plus_plus.pth"
MFTIQ_CFG = "configs/MFTIQ4_RAFT_200k_cfg.py"
CKPT_DIR = Path(r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/ckpts")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def extract_centroids(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((0, 2), dtype=np.float32)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def discover_all_seqs():
    """All 120 STIR seqs with valid GT."""
    seqs = []
    for col_dir in sorted(Path(STIR_ROOT).iterdir()):
        if not col_dir.is_dir():
            continue
        for side in ["left", "right"]:
            side_dir = col_dir / side
            if not side_dir.is_dir():
                continue
            for seq_dir in sorted(side_dir.glob("seq*")):
                vids = list((seq_dir / "frames").glob("*.mp4"))
                if not vids:
                    continue
                seqs.append({"col": col_dir.name, "side": side, "name": seq_dir.name,
                             "video": vids[0], "seg_dir": seq_dir / "segmentation",
                             "key": f"{col_dir.name}_{side}_{seq_dir.name}"})
    return seqs


def get_test_keys(per_collection=3, seed=42):
    """Replicate the same RNG-42 sampling we used for our 25-seq test."""
    rng = np.random.default_rng(seed)
    test_keys = set()
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
                if vids:
                    col_seqs.append(f"{col_dir.name}_{side}_{seq_dir.name}")
        if len(col_seqs) > per_collection:
            idx = rng.choice(len(col_seqs), per_collection, replace=False)
            col_seqs = [col_seqs[i] for i in sorted(idx)]
        test_keys.update(col_seqs)
    return test_keys


def split_train_val(seed=0, val_size=15, max_frames=None):
    """80 train + 15 val + 25 test (locked from prior eval).

    If max_frames is given, skip any train seqs with more frames than that
    (super-long seqs like 11_left_seq09 with 2006 frames hang chain backward).
    """
    all_seqs = discover_all_seqs()
    test_keys = get_test_keys()
    train_val = [s for s in all_seqs if s["key"] not in test_keys]
    if max_frames is not None:
        kept = []
        skipped = []
        for s in train_val:
            cap = cv2.VideoCapture(str(s["video"]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n <= max_frames:
                kept.append(s)
            else:
                skipped.append((s["key"], n))
        if skipped:
            print(f"[split] skipped {len(skipped)} seq(s) with > {max_frames} frames:")
            for k, n in skipped:
                print(f"        {k}  ({n} frames)")
        train_val = kept
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_val))
    val_idx = set(perm[:val_size].tolist())
    train = [train_val[i] for i in range(len(train_val)) if i not in val_idx]
    val = [train_val[i] for i in val_idx]
    return train, val, sorted(test_keys)


def load_seq(video_path, scale=0.5):
    """Read all frames, half-res. Returns numpy (N, H, W, 3) BGR uint8."""
    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_W = int(W * scale) & ~1
    new_H = int(H * scale) & ~1
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(bgr, (new_W, new_H), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames), (new_W / W, new_H / H)


def to_tensor(bgr_frame, device):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
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


def build_models(adapter_type, ckpt_path=None):
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
    else:
        adapter = CrossAttentionSpectralAdapter(n_bands=31, dim=64, n_heads=4,
                                                feat_dim=256, stride=8).cuda()

    gated = GatedFusion(orig_fnet, adapter, gamma_init=1.0,
                        ms_from_rgb=ms_fn, rgb_input_normalized=True).cuda()
    # tiny non-zero init so all params receive gradient from step 1
    with torch.no_grad():
        last = adapter.proj if adapter_type == "self_attn" else adapter.out_proj
        nn.init.normal_(last.weight, std=1e-4)
        nn.init.normal_(last.bias, std=1e-4)
    raft.fnet = gated

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cuda")
        gated.adapter.load_state_dict(ckpt["adapter"])
        gated.gamma.data.copy_(ckpt["gamma"].to(gated.gamma.device))
        print(f"  loaded ckpt step={ckpt['step']}, γ={gated.gamma.item():.4f}")

    gated.train()
    gated.base.eval()
    for p in gated.base.parameters():
        p.requires_grad_(False)
    for p in gated.adapter.parameters():
        p.requires_grad_(True)
    gated.gamma.requires_grad_(True)
    return tracker, raft, gated, mst


# ─────────────────────────────────────────────────────────────────────────────
# Position chaining
# ─────────────────────────────────────────────────────────────────────────────
def sample_flow_at(flow, points_xy):
    """Bilinear-sample flow at (N, 2) points (in pixel coords). Returns (N, 2) sampled flows."""
    _, _, H, W = flow.shape
    # normalize to [-1, 1] for grid_sample
    grid = points_xy.clone().float()
    grid[..., 0] = 2 * grid[..., 0] / max(W - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(H - 1, 1) - 1
    # grid_sample expects (B, H, W, 2). Use (1, N, 1, 2).
    grid = grid.view(1, -1, 1, 2)
    sampled = F.grid_sample(flow, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # sampled: (1, 2, N, 1) → (N, 2)
    return sampled[0, :, :, 0].t()


def raft_forward_pair(raft, img1, img2):
    """Single RAFT forward returning flow. Wrapped for gradient checkpointing."""
    out = raft(img1, img2, iters=12, test_mode=True)
    return out["flow"]


def chain_endpoint_loss(raft, frames_uint8, queries_xy, gt_end_xy, device,
                        use_checkpoint=True, progress_every=100):
    """Chain RAFT forward through all consecutive frame pairs, accumulating queries.

    Memory bound: with gradient checkpointing, one RAFT forward graph in memory at a time.
    Prints chain progress every `progress_every` frames so a long seq is visible.
    """
    import time as _t
    N = frames_uint8.shape[0]
    pos = queries_xy.clone()
    t0 = _t.time()
    for t in range(N - 1):
        img1 = to_tensor(frames_uint8[t], device)
        img2 = to_tensor(frames_uint8[t + 1], device)
        if use_checkpoint:
            flow = checkpoint(raft_forward_pair, raft, img1, img2, use_reentrant=False)
        else:
            flow = raft_forward_pair(raft, img1, img2)
        dxy = sample_flow_at(flow, pos)
        pos = pos + dxy
        if progress_every > 0 and (t + 1) % progress_every == 0:
            print(f"    chain {t + 1}/{N - 1}  elapsed {_t.time() - t0:.1f}s", flush=True)
    diff = pos - gt_end_xy
    loss = (diff ** 2).sum(dim=-1).mean()
    return loss, pos


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_endpoint(raft, seqs, scale=0.5, max_seqs=None, device="cuda", verbose=True):
    """Compute mean endpoint error on a list of seqs (no grad)."""
    import time as _time
    raft_fn = raft  # alias
    raft.eval()
    all_d = []
    n = 0
    for s in seqs:
        if max_seqs is not None and n >= max_seqs:
            break
        try:
            t_v0 = _time.time()
            start = extract_centroids(s["seg_dir"] / "icgstartseg.png")
            end = extract_centroids(s["seg_dir"] / "icgendseg.png")
            if len(start) == 0 or len(end) == 0:
                continue
            frames, (sx, sy) = load_seq(s["video"], scale=scale)
            queries = torch.from_numpy(start).float().cuda()
            queries[:, 0] *= sx
            queries[:, 1] *= sy
            gt_end_small = torch.from_numpy(end).float().cuda()
            gt_end_small[:, 0] *= sx
            gt_end_small[:, 1] *= sy
            with torch.no_grad():
                pos = queries.clone()
                for t in range(len(frames) - 1):
                    img1 = to_tensor(frames[t], device)
                    img2 = to_tensor(frames[t + 1], device)
                    flow = raft_fn(img1, img2, iters=12, test_mode=True)["flow"]
                    pos = pos + sample_flow_at(flow, pos)
            # Hungarian matching (simple bipartite via scipy)
            from scipy.optimize import linear_sum_assignment
            pred_np = pos.cpu().numpy()
            end_np = gt_end_small.cpu().numpy()
            D = np.linalg.norm(pred_np[:, None] - end_np[None, :], axis=-1)
            if D.shape[0] != D.shape[1]:
                big = max(D.shape) + 1
                Dp = np.full((big, big), 1e9, dtype=np.float64)
                Dp[:D.shape[0], :D.shape[1]] = D
                D = Dp
            pi, gi = linear_sum_assignment(D)
            d_this = []
            for p, g in zip(pi, gi):
                if p < len(pred_np) and g < len(end_np):
                    # scale-back to original resolution
                    d_full = np.linalg.norm((pred_np[p] / np.array([sx, sy])) - (end_np[g] / np.array([sx, sy])))
                    d_this.append(d_full)
            all_d.extend(d_this)
            n += 1
            if verbose:
                med = np.median(d_this) if d_this else float("nan")
                print(f"    val [{n}/{max_seqs if max_seqs else len(seqs)}] {s['key']:30s} "
                      f"frames={len(frames):3d} n_pts={len(d_this):2d} "
                      f"median={med:6.2f}px  {(_time.time() - t_v0):.1f}s")
            # explicit cleanup to keep CPU RAM bounded
            del frames
        except Exception as e:
            print(f"eval failed on {s['key']}: {e}")
    raft.train()
    return np.array(all_d), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", type=str, default="cross_attn",
                        choices=["self_attn", "cross_attn"])
    parser.add_argument("--init_ckpt", type=str, default=None, help="warm-start adapter from ckpt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--val_every", type=int, default=1, help="run val every N epochs")
    parser.add_argument("--val_max_seqs", type=int, default=15,
                        help="how many val seqs (0=all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="supA1")
    parser.add_argument("--no_checkpoint", action="store_true", help="disable gradient checkpointing")
    parser.add_argument("--max_train_seqs", type=int, default=0,
                        help="if >0, use only first N train seqs (smoke test)")
    parser.add_argument("--max_frames", type=int, default=600,
                        help="skip seqs longer than this many frames (avoid hang on huge seqs)")
    parser.add_argument("--ckpt_every_steps", type=int, default=20,
                        help="save adapter ckpt every N train steps (in addition to end-of-epoch)")
    parser.add_argument("--progress_every", type=int, default=100,
                        help="print chain progress every N frames inside a step")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CKPT_DIR / f"sup_{args.tag}.log.jsonl"

    print(f"[setup] building models ({args.adapter_type}) ...")
    tracker, raft, gated, mst = build_models(args.adapter_type, ckpt_path=args.init_ckpt)
    n_train_p = sum(p.numel() for p in gated.adapter.parameters()) + 1
    print(f"[setup] trainable params: {n_train_p / 1e6:.3f} M")

    train_seqs, val_seqs, test_keys = split_train_val(seed=args.seed, val_size=15,
                                                       max_frames=args.max_frames)
    if args.max_train_seqs > 0:
        train_seqs = train_seqs[:args.max_train_seqs]
    print(f"[setup] split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_keys)}")

    opt = torch.optim.AdamW(
        list(gated.adapter.parameters()) + [gated.gamma],
        lr=args.lr, weight_decay=1e-5,
    )

    use_ckpt = not args.no_checkpoint
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    with open(log_path, "w") as flog:
        for epoch in range(args.epochs):
            print(f"\n=== epoch {epoch + 1}/{args.epochs} ===")
            perm = rng.permutation(len(train_seqs))
            ep_losses = []
            for step_i, seq_idx in enumerate(perm):
                s = train_seqs[seq_idx]
                try:
                    start = extract_centroids(s["seg_dir"] / "icgstartseg.png")
                    end = extract_centroids(s["seg_dir"] / "icgendseg.png")
                    if len(start) == 0 or len(end) == 0:
                        continue
                    # Hungarian-match start and end CENTROIDS to make supervision pairs
                    from scipy.optimize import linear_sum_assignment
                    if len(start) != len(end):
                        # match by nearest pair (best-effort); skip if very different counts
                        nmin = min(len(start), len(end))
                        D = np.linalg.norm(start[:, None] - end[None, :], axis=-1)
                        if D.shape[0] != D.shape[1]:
                            big = max(D.shape) + 1
                            Dp = np.full((big, big), 1e9, dtype=np.float64)
                            Dp[:D.shape[0], :D.shape[1]] = D
                            D = Dp
                        pi, gi = linear_sum_assignment(D)
                        keep_s, keep_e = [], []
                        for p, g in zip(pi, gi):
                            if p < len(start) and g < len(end):
                                keep_s.append(start[p])
                                keep_e.append(end[g])
                        start = np.array(keep_s)
                        end = np.array(keep_e)
                    else:
                        D = np.linalg.norm(start[:, None] - end[None, :], axis=-1)
                        pi, gi = linear_sum_assignment(D)
                        start = start[pi]
                        end = end[gi]
                    frames, (sx, sy) = load_seq(s["video"], scale=args.scale)
                    queries = torch.from_numpy(start).float().cuda()
                    queries[:, 0] *= sx
                    queries[:, 1] *= sy
                    gt_end = torch.from_numpy(end).float().cuda()
                    gt_end[:, 0] *= sx
                    gt_end[:, 1] *= sy

                    opt.zero_grad(set_to_none=True)
                    t0 = time.time()
                    loss, pos = chain_endpoint_loss(raft, frames, queries, gt_end, "cuda",
                                                    use_checkpoint=use_ckpt,
                                                    progress_every=args.progress_every)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(gated.adapter.parameters()) + [gated.gamma], max_norm=1.0)
                    opt.step()
                    torch.cuda.synchronize()
                    step_t = time.time() - t0

                    ep_losses.append(loss.item())
                    rec = {"epoch": epoch, "step": step_i, "seq": s["key"],
                           "n_frames": int(len(frames)), "n_pts": int(start.shape[0]),
                           "loss": loss.item(), "gamma": float(gated.gamma.item()),
                           "step_time": step_t,
                           "elapsed_min": (time.time() - t_start) / 60}
                    flog.write(json.dumps(rec) + "\n")
                    flog.flush()
                    if step_i % 5 == 0:
                        print(f"  [{step_i:3d}/{len(perm)}] {s['key']:30s}  "
                              f"frames={len(frames):3d} n_pts={start.shape[0]:2d}  "
                              f"loss={loss.item():.2f}  γ={gated.gamma.item():.4f}  {step_t:.1f}s")
                    # save intermediate ckpt every N steps (resilient to mid-epoch hang)
                    global_step = epoch * len(perm) + step_i + 1
                    if args.ckpt_every_steps > 0 and global_step % args.ckpt_every_steps == 0:
                        ckpt = {"step": global_step, "epoch": epoch + 1,
                                "adapter": gated.adapter.state_dict(),
                                "gamma": gated.gamma.detach().cpu(),
                                "adapter_type": args.adapter_type,
                                "args": vars(args)}
                        torch.save(ckpt, CKPT_DIR / f"{args.tag}_step{global_step:06d}.pt")
                        print(f"  [ckpt] {args.tag}_step{global_step:06d}.pt", flush=True)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"OOM on {s['key']} (frames={len(frames) if 'frames' in dir() else '?'}), skipping")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"failed on {s['key']}: {e}")
                    traceback.print_exc()

            mean_ep_loss = np.mean(ep_losses) if ep_losses else float("nan")
            print(f"  epoch {epoch + 1} mean train loss: {mean_ep_loss:.2f}")

            # Save ckpt FIRST so we never lose epoch work if val hangs
            ckpt = {"step": (epoch + 1) * len(perm), "epoch": epoch + 1,
                    "adapter": gated.adapter.state_dict(),
                    "gamma": gated.gamma.detach().cpu(),
                    "adapter_type": args.adapter_type,
                    "args": vars(args)}
            torch.save(ckpt, CKPT_DIR / f"{args.tag}_epoch{epoch + 1:02d}.pt")
            print(f"  [ckpt] {args.tag}_epoch{epoch + 1:02d}.pt")

            # Validation (with per-seq progress)
            if (epoch + 1) % args.val_every == 0 and args.val_max_seqs > 0:
                print(f"  validation on {args.val_max_seqs} seqs ...")
                gated.adapter.eval()
                with torch.no_grad():
                    dists, n = evaluate_endpoint(raft, val_seqs, scale=args.scale,
                                                 max_seqs=args.val_max_seqs)
                gated.adapter.train()
                if len(dists):
                    print(f"  val ({n} seqs, {len(dists)} pts): median={np.median(dists):.2f}px  "
                          f"acc@4/8/16/32/64={[round(float((dists<t).mean()),3) for t in [4,8,16,32,64]]}  "
                          f"avg={np.mean([(dists<t).mean() for t in [4,8,16,32,64]]):.4f}")

    print(f"\n[done] total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
