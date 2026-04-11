# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Masked Autoencoder (MAE) for self-supervised pretraining on Yiddish OCR text-line images. Grayscale 32x512px images, ViT-Base encoder with patch size 8. Based on *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2022).

## Training

```bash
# Single GPU
python train.py
```

Training config is in `train.py` (no argparse). Dataset images go in `./data/yiddish_lines/`.

TensorBoard: `tensorboard --logdir ./runs/mae_yiddish`

Profiler traces: `tensorboard --logdir ./runs/profile`

## Architecture

```
mae/
├── model.py          # MaskedAutoencoderViT + FlashAttention + factory functions
├── dataset.py        # YiddishSharedInRamDataset (loads all images into shared memory)
└── pos_embed.py      # 2D sin-cos positional embeddings (supports rectangular grids)
train.py              # training entry point (torch.compile + torch.autocast bf16)
training_logger.py    # TrainingLogger — rich progress bar, VRAM tracking, profiler, summary
```

**MAE flow:** image -> patch embed -> random mask 75% -> encode visible patches only -> decoder re-inserts mask tokens -> reconstruct full image -> MSE loss on masked patches only.

**FlashAttention** wraps `F.scaled_dot_product_attention` and is injected into timm `Block` via `attn_layer=FlashAttention`.

**Factory functions** in `mae/model.py`: `mae_vit_base_patch16`, `mae_vit_large_patch16`, `mae_vit_huge_patch14`, `mae_vit_base_patch8_32x512` (Yiddish config).

**Mixed precision** via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — runs matmuls/convs in bf16, keeps numerically sensitive ops (softmax, layer norm, loss) in fp32 automatically.

## Key Hyperparameters

- **mask_ratio=0.75** (75% patches masked, 25% visible to encoder)
- **norm_pix_loss=True** normalizes target patches to zero-mean unit-variance before MSE
- **batch_size=256**, **lr=1.5e-4**, **weight_decay=0.05**

## TrainingLogger

`TrainingLogger` in `training_logger.py` separates all visualization from training logic. Interface:

```python
with TrainingLogger(device, num_epochs, len(dataloader), TENSORBOARD_PROFILE) as logger:
    logger.begin_epoch(epoch)   # resets progress bar
    logger.on_step(loss.item()) # updates bar, tracks VRAM, steps profiler
    logger.end_epoch(epoch)     # prints epoch summary
```

Tracks: rich live progress bar, per-epoch summaries, peak VRAM, avg loss, steps/sec. Profiler runs for first 5 steps only (wait=1, warmup=1, active=3) then stops automatically.

Set `TENSORBOARD_PROFILE` in `train.py` to a unique name per experiment to compare runs side-by-side in TensorBoard.

## Performance

Target: RTX 3090 (24GB). Optimizations: bf16 autocast, torch.compile(mode="reduce-overhead"), FlashAttention via SDPA, in-RAM dataset with shared memory, persistent workers.
