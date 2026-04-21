# Training Run 1 — Dual GPU (2× RTX 3090), 50 Epochs

**Date:** 2026-04-21
**Config:** `mae_vit_ultra_light`, 2× RTX 3090, `torchrun --nproc_per_node=2`
**Batch:** 9216 per GPU (18432 total), lr=9.6e-3 (linear scaled), 50 epochs

## Loss Curve

| Epoch | Loss | Notes |
|-------|------|-------|
| 1 | 0.9571 | 20.9s — CUDA graph compilation warmup |
| 2 | 0.7428 | |
| 3 | 0.7136 | |
| 4–7 | 0.708 → 0.705 | Fast initial descent |
| 8 | 0.7045 | **Plateau begins** |
| 18–19 | 0.7059 | Spike — lr instability |
| 24 | 0.7093 | Spike — lr instability |
| 50 | 0.7032 | Final — negligible improvement from epoch 8 |

## Speed

- Epoch 1: 20.9s (CUDA graph capture + torch.compile autotune)
- Epochs 2–50: **3.1–3.8s/epoch** — DDP working correctly

## Problems Identified

### 1. Model plateaued at epoch 8 — no further learning

Total improvement epochs 8→50: **0.0013** (effectively zero).

### 2. Too few gradient steps per epoch

At 3.5s/epoch and ~10ms/GPU-step, there are only **~2–4 gradient updates per epoch**.
With 50 epochs that is ~100–200 total optimizer steps — far too few for meaningful training.

`OneCycleLR` configured for those 100 steps runs its full warmup→cosine-decay cycle in
~100 steps. The lr reaches near-zero by epoch 50, but the model barely learned.

Verify with:
```bash
python -c "
from mae.dali_loader import build_dali_loader
dl = build_dali_loader('./data/yiddish_lines.npy', batch_size=9216, num_threads=4)
print('steps/epoch per GPU:', len(dl))
"
```

### 3. LR too high → loss spikes

lr=9.6e-3 (4.8e-3 × 2 GPUs, linear scaling rule) caused visible instability spikes
at epochs 18–19 and 23–24. Suggests 9.6e-3 is above the stable range for this model/dataset.

### 4. Possible model capacity floor at ~0.70

`mae_vit_ultra_light` (5.17M params) may genuinely lack capacity to reconstruct
masked Yiddish text patches below ~0.70 with `norm_pix_loss=True`.
Reference: random predictor ≈ 1.0; 0.70 = 30% better than random.

## Recommendations

| Issue | Fix |
|-------|-----|
| Too few steps/epoch | Run `--epochs 500+` or switch to step-based total_steps |
| LR too high | Revert to 4.8e-3 flat (do not scale by world_size for small datasets) |
| Model capacity floor | Switch to `mae_vit_base_patch32x8_32x512` (86M params) |

## Next Steps

1. Check `len(dataloader)` to confirm steps/epoch is in single digits
2. If yes: re-run with `--epochs 500` and lr=4.8e-3
3. Monitor whether loss continues below 0.70 with more steps
4. If still stuck at 0.70: upgrade model size
