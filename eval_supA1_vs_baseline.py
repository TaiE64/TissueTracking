"""Compare A1-trained adapter vs baseline (γ=0) on the SAME 5 val seqs used during training.

This avoids the 3-4 h full STIR eval. We use the same chained-RAFT evaluation
that the training loop uses (raft.forward per frame pair) — internally consistent.

Pass 1: γ=0 (adapter effectively bypassed) — baseline behavior
Pass 2: trained adapter from supA1_epoch01.pt
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
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from train_supervised import (
    build_models, split_train_val, evaluate_endpoint, discover_all_seqs,
)

CKPT = r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/ckpts/supA1_epoch01.pt"


def run_pass(label, bypass_adapter, raft, gated, eval_seqs, max_seqs=None):
    print(f"\n== {label} ==")
    if bypass_adapter:
        # γ=0 → ΔF doesn't contribute → equivalent to baseline RGB-only behavior
        gated.gamma.data.fill_(0.0)
    dists, n = evaluate_endpoint(raft, eval_seqs, scale=0.5, max_seqs=None, verbose=True)
    if len(dists) == 0:
        print("  no dists collected!")
        return None
    THRESH = [4, 8, 16, 32, 64]
    accs = [(dists < t).mean() for t in THRESH]
    print(f"  pooled {len(dists)} pts:  median={np.median(dists):.2f}px  "
          f"acc@{THRESH}={[round(float(a),3) for a in accs]}  avg={np.mean(accs):.4f}")
    return {"n_pts": len(dists), "median": float(np.median(dists)),
            "acc": {t: float((dists < t).mean()) for t in THRESH},
            "avg_acc": float(np.mean(accs)),
            "dists": dists.tolist()}


# Build model + load supA1_epoch01 ckpt
print("[setup] building tracker with cross_attn adapter ...")
tracker, raft, gated, mst = build_models("cross_attn", ckpt_path=CKPT)
gated.eval()

train_seqs, val_seqs, test_keys = split_train_val(seed=0, val_size=15, max_frames=600)
all_seqs = discover_all_seqs()
test_seqs = [s for s in all_seqs if s["key"] in test_keys]
# Use first 10 of test seqs for quick comparison
test_seqs = test_seqs[:10]
print(f"[setup] eval on {len(test_seqs)} TEST seqs (subset of the 25 reserved)")

trained_gamma = gated.gamma.item()
print(f"[setup] trained γ = {trained_gamma:.4f}")

# Pass 1: bypass adapter (γ=0) → baseline
baseline_result = run_pass("baseline (γ=0)", bypass_adapter=True, raft=raft, gated=gated, eval_seqs=test_seqs)

# Reset γ to trained value for pass 2
gated.gamma.data.fill_(trained_gamma)

# Pass 2: trained adapter
adapter_result = run_pass(f"trained adapter (γ={trained_gamma:.4f})", bypass_adapter=False, raft=raft, gated=gated, eval_seqs=test_seqs)

# Summary
print("\n=" * 60)
print("FINAL COMPARISON (5 val seqs, chained RAFT eval):")
print(f"{'method':30s} {'n_pts':>6s} {'median':>8s} {'acc@4':>7s} {'acc@8':>7s} {'acc@16':>7s} {'acc@32':>7s} {'acc@64':>7s} {'avg':>9s}")
for label, r in [("baseline (γ=0, RGB-only)", baseline_result), ("trained adapter (supA1)", adapter_result)]:
    if r is None:
        continue
    accs = [r['acc'][t] for t in [4, 8, 16, 32, 64]]
    print(f"{label:30s} {r['n_pts']:>6d} {r['median']:>8.2f} {accs[0]:>7.3f} {accs[1]:>7.3f} {accs[2]:>7.3f} {accs[3]:>7.3f} {accs[4]:>7.3f} {r['avg_acc']:>9.4f}")
delta = adapter_result['avg_acc'] - baseline_result['avg_acc']
print(f"\nΔ avg acc = {delta:+.4f}  ({'adapter better' if delta > 0 else 'baseline better' if delta < 0 else 'tied'})")

# Save
with open(r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out/zero_shot_eval/supA1_vs_baseline_val5.json", "w") as f:
    json.dump({"baseline": baseline_result, "adapter_supA1_epoch01": adapter_result,
               "delta_avg_acc": delta, "val_seqs": [s["key"] for s in val_seqs[:5]]}, f, indent=2)
