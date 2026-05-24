"""Quick verification of CrossAttentionSpectralAdapter.

Checks:
  1. Forward produces correct output shape
  2. Zero-init out_proj → ΔF = 0 → F_fused == F_rgb bit-exact
  3. After random init of out_proj → output differs (signal flows)
  4. Backward works → adapter gets gradient
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

import sys
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/mst_stir_out")
sys.path.insert(0, r"c:/Users/29421/Desktop/TissueTracking/MFTIQ-master/MFTIQ-master")

import torch
import torch.nn as nn
from MFTIQ.RAFT.core.extractor import BasicEncoder
from spectral_adapter import CrossAttentionSpectralAdapter, GatedFusion


def main():
    torch.manual_seed(0)
    device = "cuda"
    B, H, W = 1, 512, 640
    feat_dim = 256

    fnet = BasicEncoder(output_dim=feat_dim, norm_fn="instance", dropout=0.0).to(device).eval()
    for p in fnet.parameters():
        p.requires_grad_(False)

    adapter = CrossAttentionSpectralAdapter(
        n_bands=31, dim=64, n_heads=4, feat_dim=feat_dim, stride=8
    ).to(device).eval()
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"CrossAttention adapter params: {n_params / 1e6:.3f} M")

    # 1. Forward shape
    rgb = torch.randn(B, 3, H, W, device=device)
    ms = torch.randn(B, 31, H, W, device=device)
    with torch.no_grad():
        F_rgb = fnet(rgb)
        dF = adapter(ms, F_rgb)
    print(f"\nF_rgb shape: {F_rgb.shape}")
    print(f"ΔF shape:    {dF.shape}")
    assert F_rgb.shape == dF.shape, "shape mismatch"
    print(f"  PASS: shapes match")

    # 2. Zero-init equivalence via GatedFusion
    wrap = GatedFusion(fnet, adapter, gamma_init=1.0).to(device).eval()
    with torch.no_grad():
        # Drive forward via wrap. ms=None is fine since we'll pass ms manually
        F_fused = wrap(rgb, ms)
    diff = (F_fused - F_rgb).abs().max().item()
    print(f"\n[zero-init equivalence]")
    print(f"  max abs diff (F_fused vs F_rgb): {diff:.6e}")
    assert diff == 0.0, f"zero-init must be bit-exact, got {diff}"
    print(f"  PASS: F_fused == F_rgb at zero-init (bit-exact)")

    # 3. Perturb out_proj → output should differ
    with torch.no_grad():
        nn.init.normal_(adapter.out_proj.weight, std=0.02)
        nn.init.normal_(adapter.out_proj.bias, std=0.02)
        F_fused_p = wrap(rgb, ms)
    pert_diff = (F_fused_p - F_rgb).abs().max().item()
    print(f"\n[perturbation]")
    print(f"  max abs diff after random out_proj: {pert_diff:.4e}")
    assert pert_diff > 1e-3, "non-zero adapter should produce non-trivial perturbation"
    print(f"  PASS: spectral signal flows")

    # 4. Backward: gradient reaches every adapter param
    wrap.train()
    for p in wrap.base.parameters():
        p.requires_grad_(False)
    for p in wrap.adapter.parameters():
        p.requires_grad_(True)
    wrap.gamma.requires_grad_(True)

    F_fused = wrap(rgb, ms)
    loss = F_fused.pow(2).mean()
    loss.backward()

    grad_summary = {}
    for name, p in adapter.named_parameters():
        if p.grad is None:
            grad_summary[name] = "NO GRAD"
        else:
            grad_summary[name] = f"|g|.sum = {p.grad.abs().sum().item():.4f}"
    print(f"\n[gradient flow]")
    for n, s in grad_summary.items():
        print(f"  {n:32s}  {s}")
    print(f"  γ.grad: {wrap.gamma.grad.item():.4e}")
    none_grads = [n for n, p in adapter.named_parameters() if p.grad is None]
    assert not none_grads, f"params with NO grad: {none_grads}"
    print(f"  PASS: all adapter params receive gradient")

    # 5. Two-frame path (RAFT-style list)
    wrap.eval()
    rgb_list = [rgb, rgb + 0.01 * torch.randn_like(rgb)]
    ms_list = [ms, torch.randn_like(ms)]
    with torch.no_grad():
        F_fused_tuple = wrap(rgb_list, ms_list)
    assert isinstance(F_fused_tuple, tuple) and len(F_fused_tuple) == 2
    print(f"\n[two-frame]: output tuple of {len(F_fused_tuple)}, each shape {F_fused_tuple[0].shape}")
    print(f"  PASS")

    # 6. Memory at half-res
    print(f"\n[memory at half STIR resolution 640x512]")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        F_fused = wrap(rgb, ms)
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
