"""Feature-level spectral fusion adapter for MFTIQ.

Design:
  RGB ──→ frozen MFTIQ encoder (fnet/cnet) ───→ F_rgb (B, 256, H/8, W/8)
                                                   │
  MS  ──→ SpectralAdapter ────────────────────→ ΔF_spec (B, 256, H/8, W/8)
                                                   │
                              F_fused = F_rgb + γ · ΔF_spec   (γ learnable, init 0)

Key properties:
- MFTIQ encoder is left bit-exact (no conv1 surgery, no pretrained weight loss)
- γ=0 at init → F_fused == F_rgb exactly (no fp drift)
- Spectral path is gated: adapter or γ can independently turn it off
- All new params (adapter + γ) are trainable; MFTIQ stays frozen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttentionTrunk(nn.Module):
    """Per-pixel spectral self-attention block.
    (B, n_bands, H, W) -> (B, dim, H, W)
    """
    def __init__(self, n_bands=31, dim=32, n_heads=4, n_layers=3, group_size=3,
                 chunk_size=16384):
        super().__init__()
        self.n_bands = n_bands
        self.dim = dim
        self.chunk_size = chunk_size
        self.gse = nn.Conv1d(1, dim, kernel_size=group_size, padding=group_size // 2)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads,
                                       dim_feedforward=2 * dim,
                                       batch_first=True, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.caf = nn.Parameter(torch.zeros(n_layers))  # CAF: zero-init skip weights

    def forward(self, ms):
        B, C, H, W = ms.shape
        N = B * H * W
        x_flat = ms.permute(0, 2, 3, 1).reshape(N, 1, C)
        out_flat = torch.empty(N, self.dim, device=ms.device, dtype=ms.dtype)
        for i in range(0, N, self.chunk_size):
            xc = x_flat[i:i + self.chunk_size]
            h = self.gse(xc).transpose(1, 2)  # (chunk, C, dim)
            prev = None
            for j, layer in enumerate(self.layers):
                h = layer(h)
                if prev is not None:
                    h = h + self.caf[j] * prev
                prev = h
            out_flat[i:i + self.chunk_size] = h.mean(dim=1)  # pool over band axis
        return out_flat.view(B, H, W, self.dim).permute(0, 3, 1, 2).contiguous()


class FeatureLevelSpectralAdapter(nn.Module):
    """Produce a spectral feature correction matched to MFTIQ's RAFT fnet output shape.

    out_shape: (B, feat_dim, H/stride, W/stride)

    Order:
      1. spatial downsample MS first via strided pixel-unshuffle-like convs
         (makes per-pixel attention 60x cheaper and avoids OOM in backward)
      2. spectral self-attention over band axis at the downsampled grid
      3. final 1x1 conv (zero-init) to feat_dim
    """
    def __init__(self, n_bands=31, trunk_dim=32, n_layers=3,
                 feat_dim=256, stride=8, group_size=3):
        super().__init__()
        self.n_bands = n_bands
        # Spatially downsample MS by `stride` BEFORE spectral attention.
        # We keep n_bands channels through the downsampling using depthwise/grouped convs
        # so each band's spatial info is summarized independently. Then merge at attention stage.
        log2_stride = stride.bit_length() - 1
        # depthwise (groups=n_bands) keeps bands separate during spatial pool
        dws = []
        cur_c = n_bands
        for _ in range(log2_stride):
            dws += [
                nn.Conv2d(cur_c, cur_c, kernel_size=3, stride=2, padding=1, groups=cur_c, bias=False),
                nn.GroupNorm(1, cur_c),
                nn.GELU(),
            ]
        self.spatial_down = nn.Sequential(*dws)
        # Spectral attention trunk runs on the downsampled grid (much fewer pixels)
        self.trunk = SpectralAttentionTrunk(n_bands=n_bands, dim=trunk_dim,
                                            n_layers=n_layers, group_size=group_size,
                                            chunk_size=16384)
        # final 1x1 conv from trunk_dim -> feat_dim (zero-init for residual gating)
        self.proj = nn.Conv2d(trunk_dim, feat_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, ms):
        # 1. spatial downsample (cheap, depthwise)
        x = self.spatial_down(ms)  # (B, n_bands, H/stride, W/stride)
        # 2. spectral self-attention at low res (cheap & backprop-friendly)
        f = self.trunk(x)  # (B, trunk_dim, H/stride, W/stride)
        # 3. project to feat_dim
        return self.proj(f)


class GatedFusion(nn.Module):
    """Wrap a frozen feature extractor (e.g., RAFT fnet) and inject a learnable
    additive spectral correction: F_fused = F_rgb + γ · ΔF_spec.

    IMPORTANT init convention for TRAINING (do NOT use γ=0 + head_zero simultaneously):
      - γ_init = 1.0
      - adapter head final 1×1 conv weight = 0 (so ΔF_spec = 0 at t=0)
      With this combo:
        * At t=0: γ·ΔF = 1·0 = 0  →  F_fused ≡ F_rgb  (exact equivalence)
        * ∂L/∂head_w = (∂L/∂F_fused)·γ·∂ΔF/∂w  with γ=1  →  head receives gradient
        * ∂L/∂γ = (∂L/∂F_fused)·ΔF  starts at 0 but becomes nonzero once head moves.

      Setting γ=0 AND head_zero at the same time makes BOTH gradients zero
      forever (no signal to break the symmetry). For ablation only, set γ=0
      and run forward (no backward).

    The forward accepts either a single tensor (rgb shape (B,3,H,W)) or a
    list/tuple of two tensors (RAFT-style stereo / two-frame batch). The
    adapter is applied to BOTH MS frames when ms is a list — the two-frame
    application is required so the correlation volume is computed between
    same-distribution features.
    """
    def __init__(self, base_encoder, adapter, gamma_init=1.0, ms_from_rgb=None,
                 rgb_input_normalized=True):
        super().__init__()
        self.base = base_encoder
        self.adapter = adapter
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # ms_from_rgb: optional callable rgb -> ms. Used when forward is called
        # without an explicit ms arg (e.g., RAFT.forward calls fnet([img1, img2])
        # with no MS). Should return a (B, n_bands, H, W) tensor.
        self.ms_from_rgb = ms_from_rgb
        # rgb_input_normalized: True if the rgb tensors entering forward are in
        # RAFT's normalized [-1, 1] range; the function will undo to [0, 1] before
        # calling ms_from_rgb (MST++ expects [0, 1]).
        self.rgb_input_normalized = rgb_input_normalized

    def _derive_ms_for(self, rgb_tensor):
        """Given an rgb tensor (B, 3, H, W) in either [-1,1] or [0,1],
        produce a ms tensor via self.ms_from_rgb (typically a frozen MST++).
        The MS is detached so MST++ doesn't accumulate gradient (kept frozen)."""
        if self.rgb_input_normalized:
            rgb01 = (rgb_tensor + 1.0) / 2.0
        else:
            rgb01 = rgb_tensor
        rgb01 = rgb01.clamp(0.0, 1.0)
        with torch.no_grad():
            return self.ms_from_rgb(rgb01).detach()

    def _adapter_paired(self, ms_pair):
        """Apply adapter to each ms in a pair and cat along batch dim.
        Matches how RAFT's BasicEncoder concats [image1, image2] internally.
        """
        # cat along batch then call adapter once (efficient) OR call per-frame
        # We call per-frame so each frame's spectral attention is independent.
        return torch.cat([self.adapter(m) for m in ms_pair], dim=0)

    def forward(self, rgb, ms=None):
        F_rgb = self.base(rgb)

        # If MS not explicitly given but we have a derivation hook (e.g., MST++),
        # compute MS from the same RGB tensor(s) on the fly.
        if ms is None and self.ms_from_rgb is not None:
            if isinstance(rgb, (list, tuple)):
                ms = [self._derive_ms_for(r) for r in rgb]
            else:
                ms = self._derive_ms_for(rgb)

        if ms is None:
            return F_rgb

        if isinstance(ms, (list, tuple)):
            assert isinstance(F_rgb, tuple) and len(F_rgb) == len(ms), \
                "ms must be a sequence with same length as the base's tuple output"
            # apply adapter to EACH frame; addition is per-frame to its matching F_rgb
            return tuple(f + self.gamma * self.adapter(m) for f, m in zip(F_rgb, ms))

        dF = self.adapter(ms)
        if isinstance(F_rgb, tuple):
            # shouldn't normally happen (single rgb -> tuple) but guard anyway
            return tuple(f + self.gamma * dF for f in F_rgb)
        return F_rgb + self.gamma * dF
