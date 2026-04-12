# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Masked Autoencoder (MAE) for self-supervised pretraining on Yiddish OCR text-line images. Grayscale 32x512px images, ViT-Base encoder with patch size 8. Based on *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2022).

## Training

```bash
# Prepare dataset once (decode + resize all images into a single memmap file)
python prepare_dataset.py

# Single GPU
python train.py
```

Training config is in `train.py` (no argparse). Dataset images go in `./data/yiddish_lines/`.
The prepared memmap is written to `./data/yiddish_lines.npy`.

TensorBoard: `tensorboard --logdir ./runs/mae_yiddish`

Profiler traces: `tensorboard --logdir ./runs/<TENSORBOARD_PROFILE>`

## Architecture

```
mae/
├── model.py          # MaskedAutoencoderViT + FlashAttention + factory functions
├── dali_loader.py    # build_dali_loader — NVIDIA DALI pipeline backed by numpy memmap
├── dataset.py        # YiddishSharedInRamDataset (legacy, kept for reference)
└── pos_embed.py      # 2D sin-cos positional embeddings (supports rectangular grids)
train.py              # training entry point (torch.compile + torch.autocast bf16)
prepare_dataset.py    # one-time script: decode + resize images → ./data/yiddish_lines.npy
training_logger.py    # TrainingLogger — rich progress bar, VRAM tracking, profiler, summary
analyze_trace.py      # CLI tool: analyze PyTorch profiler .pt.trace.json files
```

**MAE flow:** image -> patch embed -> random mask 75% -> encode visible patches only -> decoder re-inserts mask tokens -> reconstruct full image -> MSE loss on masked patches only.

**FlashAttention** wraps `F.scaled_dot_product_attention` and is injected into timm `Block` via `attn_layer=FlashAttention`.

**Factory functions** in `mae/model.py`: `mae_vit_base_patch16`, `mae_vit_large_patch16`, `mae_vit_huge_patch14`, `mae_vit_base_patch8_32x512` (Yiddish config).

**Mixed precision** via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — runs matmuls/convs in bf16, keeps numerically sensitive ops (softmax, layer norm, loss) in fp32 automatically.

## Key Hyperparameters

- **mask_ratio=0.75** (75% patches masked, 25% visible to encoder)
- **norm_pix_loss=True** normalizes target patches to zero-mean unit-variance before MSE
- **batch_size=256**, **lr=1.5e-4**, **weight_decay=0.05**

## Data Pipeline

`prepare_dataset.py` runs once: decodes all images with PIL, resizes to 32×512, saves as
a single `(N, 32, 512)` uint8 numpy memmap (`./data/yiddish_lines.npy`). Decode and resize
cost is paid once, not every epoch.

`mae/dali_loader.py` — `build_dali_loader(npy_path, batch_size, num_threads, device_id)`:
- `np.load(..., mmap_mode='r')` memory-maps the file; OS manages page cache
- `fn.external_source` feeds samples to DALI, reshuffled each epoch
- `fn.crop_mirror_normalize` converts uint8 HWC → float32 CHW, normalizes to [0, 1] on GPU
- Returns a `DALIGenericIterator`; batches are already on GPU as `(N, 1, H, W)` float32
- Iterate: `batch = batch_data[0]["images"]` — no `.to()`, `.float()`, or `.div_()` needed

## TrainingLogger

`TrainingLogger` in `training_logger.py` separates all visualization from training logic. Interface:

```python
with TrainingLogger(device, num_epochs, len(dataloader), TENSORBOARD_PROFILE) as logger:
    logger.begin_epoch(epoch)          # resets progress bar
    logger.on_step()                   # advances bar, tracks VRAM, steps profiler
    logger.end_epoch(epoch, avg_loss)  # syncs loss once, prints epoch summary
```

Loss is accumulated on GPU across steps (`epoch_loss += loss.detach()`) and synced to CPU
once per epoch via `.item()` — avoids per-step `cudaStreamSynchronize`.

Tracks: rich live progress bar, per-epoch summaries, peak VRAM, avg loss, steps/sec. Profiler runs for first 5 steps only (wait=1, warmup=1, active=3) then stops automatically.

Set `TENSORBOARD_PROFILE` in `train.py` to a unique name per experiment to compare runs side-by-side in TensorBoard.

## Performance

Target: RTX 3090 (24GB). Optimizations:
- **bf16 autocast** — matmuls/convs in bf16
- **`torch.compile(mode="max-autotune")`** — benchmarks kernel configs on first run, caches to `.cache/`; subsequent runs load from cache
- **Full train_step compiled** — `zero_grad` + forward + backward + optimizer captured into two CUDA graphs; CPU only fires two `cudaGraphLaunch` calls per step (~1ms total CPU overhead)
- **`torch.profiler.record_function`** annotations inside `train_step` — preserved by `torch.compile`, visible in TensorBoard trace; zero overhead when profiler is inactive
- **FlashAttention** via SDPA
- **DALI memmap pipeline** — no per-epoch decode/resize; GPU kernel density 99.7%
- **Loss sync once per epoch** — eliminates per-step `cudaStreamSynchronize`
- **Fused AdamW** (`fused=True`) — single `multi_tensor_apply_kernel` over all parameters

Steady-state step: ~197ms GPU, two CUDA graph launches, 99.7% kernel density.
