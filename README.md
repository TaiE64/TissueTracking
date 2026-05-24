# Spectral-aware MFTIQ for Tissue Tracking — Experiment Repo

研究问题: **能否用 MST++ 重建的多光谱信号 + SpectralFormer-style attention,
zero-shot 提升 MFTIQ 在 STIR 上的 tissue tracking 性能?**

最终结论: **No.** 在当前自监督训练 + 半分辨率 setup 下,trained adapter
在 STIR 上 avg accuracy 与 RGB-only MFTIQ 持平 (~0.58)。详见 § 结果。

---

## 架构

```
RGB frame ─→ frozen MFTIQ fnet ─────────→ F_rgb (B,256,H/8,W/8)
        ├
        └─→ frozen MST++ → 31-band MS → SpectralAdapter → ΔF_spec
                                                              ↓
                              F_fused = F_rgb + γ · ΔF_spec
                                          ↓
                          frozen MFTIQ downstream → tracking output
```

- 可训参数: SpectralAdapter (~35K) + γ (1) = **0.04% 总模型**
- γ_init = 1.0, proj.weight zero-init → `F_fused ≡ F_rgb` at t=0 (bit-exact)
- 自监督训练: photometric warp + edge-aware smoothness + 0.01·distillation

## 文件

### 核心模块
- `spectral_adapter.py` — `SpectralAttentionTrunk` + `FeatureLevelSpectralAdapter` + `GatedFusion`
- `surgt_dataset.py` — SurgT 帧对 dataset (filtered to 50 standard-res clips, 89294 pairs)

### 训练 / 评估
- `train_spectral.py` — 自监督训练 (8000 step, 半分辨率)
- `run_resized_eval.py` — RGB-only baseline 评估
- `eval_trained_adapter.py` — zero-shot adapter 评估 (支持 `--max_seqs N` 部分评估)
- `prescan_gt.py` — STIR GT 扫描 util

### 验证脚本 (架构正确性证明)
- `verify_phase1.py` — standalone adapter bit-exact equivalence + gradient flow
- `verify_phase2_0a_full.py` — REAL MFTIQ 集成 + 真 MST++ MS 注入 + 扰动测试
- `verify_phase2_0b.py` — RAFT.forward 端到端梯度可达性

### 训练产物
- `ckpts/v1_step003000.pt`, `v1_step005000.pt` — 训练 ckpts (adapter weights + γ)
- `ckpts/train_v1.log.jsonl` — 训练日志 (per-step photo/smooth/distill/γ)

### 结果数据
- `resized_eval/results.json` — RGB-only baseline 在 25 STIR seqs (363 pts): avg_acc = **0.649**
- `resized_eval/comparison.png` — baseline vs orig/natural/maxspread false-RGB 对比图
- `zero_shot_eval/v1_step5000_first5_results.json` — adapter 在前 5 STIR seqs: 0.5692 (vs 同 5 seq baseline 0.5692)
- `zero_shot_eval/v1_step5000_full_partial.json` — adapter 在前 6 STIR seqs: 0.5806 (vs baseline 0.5806)
- `surgt_dataset_sanity.png` — SurgT dataloader 采样 sanity

## 关键结果

### Baseline (RGB-only MFTIQ, 25 STIR seqs, half-res, 363 pts)
| acc@4 | acc@8 | acc@16 | acc@32 | acc@64 | **avg acc** | median (px) |
|---|---|---|---|---|---|---|
| 0.372 | 0.634 | 0.708 | 0.733 | 0.799 | **0.649** | 5.4 |

### Trained adapter (SurgT 自监督, step 5000, 同 5 seq, 26 pts)
| acc@4 | acc@8 | acc@16 | acc@32 | acc@64 | **avg acc** | median (px) |
|---|---|---|---|---|---|---|
| 0.269 | 0.423 | 0.615 | 0.769 | 0.769 | **0.5692** | 9.77 |

baseline 同 5 seq = **0.5692** (完全打平)

## 已知 limitation

1. **半分辨率 eval**: 640×512 而非 STIR 原 1280×1024。坐标已 scale back,数值在
   原图像素单位,但 sub-pixel 精度丢失。
2. **训练在半分辨率**: adapter 学到的是 640×512 尺度的语义。FCN 架构理论上可
   generalize 到全分辨率,但未实测。
3. **未做 ablation**: 随机 31 通道 / HSV-Lab 3 通道 / spectral-attention-only
   等对照实验未跑,所以无法严格归因(0.04% 参数带来的微小变化是否真来自光谱)。
4. **自监督信号弱**: photo loss 早 plateau (~step 100 后基本不降),distillation
   anchor 可能把 student 拉得太死。

## 复现

```bash
# 1. 训练 (~40 min on RTX 5080)
cd MFTIQ-master/MFTIQ-master
XFORMERS_DISABLED=1 python -u mst_stir_out/train_spectral.py --steps 8000 --tag v1

# 2. 评估 baseline (~3.5 h, 仅需跑一次)
XFORMERS_DISABLED=1 python -u mst_stir_out/run_resized_eval.py

# 3. 评估 adapter (~4 h on full 25 seqs, or --max_seqs 5 for quick check)
XFORMERS_DISABLED=1 python -u mst_stir_out/eval_trained_adapter.py \
  --ckpt mst_stir_out/ckpts/v1_step005000.pt --tag v1_step5000_full
```

## 依赖

- MFTIQ (checkpoints + 修改后的 site-packages MFTIQ)
- MST++ (model_zoo/mst_plus_plus.pth)
- STIR Challenge 2024 dataset (`STIRChallenge_2024/`)
- SurgT_train 2022 dataset (`SurgT_train/train/`)
- PyTorch 2.12+cu130, RTX 5080+, XFORMERS_DISABLED=1
