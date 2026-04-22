# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Masked Autoencoder (MAE) for self-supervised pretraining on Yiddish OCR text-line images. Grayscale 32x512px images, ViT-Small encoder with patch size (32h, 8w). Based on *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2022).

## Training

```bash
# Prepare dataset once (decode + resize all images into a single memmap file)
python prepare_dataset.py

# Single GPU
python train.py

# Two GPUs (DDP)
torchrun --nproc-per-node=2 train.py

# Key flags
python train.py --epochs 200 --accum-steps 4 --target-loss 0.45
torchrun --nproc-per-node=2 train.py --resume checkpoints/checkpoint.pt

# With PyTorch profiler — captures one post-training step to runs/<name>_<timestamp>.pt.trace.json
python train.py --profile <name>

# With Nsight Systems profiler — captures full run to nsys/<name>.nsys-rep
./profile_nsys.sh <name>
```

Training config is in `train.py`. Dataset images go in `./data/yiddish_lines/`.
The prepared memmap is written to `./data/yiddish_lines.npy`.

TensorBoard: `tensorboard --logdir ./runs/mae_yiddish`

## Hyperparameter Search (HPO)

Optuna + PyTorch Lightning. Searches: optimizer, lr, weight_decay, mask_ratio, betas/momentum.

```bash
# Phase 1 — broad scan (all 7 optimizers, 30 trials × 8 epochs)
./hpo_scripts/phase1_broad.sh

# Phase 2 — fine search locked to winning optimizer
./hpo_scripts/phase2_fine.sh adamw   # replace with phase 1 winner

# Show best parameters found
./hpo_scripts/phase3_show_best.sh
```

Results persist in `hpo.db` (SQLite). Safe to Ctrl-C and resume.

Optimizers searched: `adamw`, `adam`, `adabelief`, `radam`, `lion`, `sgd`, `madgrad`.
Adam-family uses betas; SGD-family uses momentum (+ Nesterov for SGD only).

```bash
# Run directly with custom options
python hpo.py --n-trials 50 --n-epochs 15 --device 1 --storage sqlite:///hpo.db
python hpo.py --optimizer adamw --n-trials 60  # phase 2 manually
```

## Finding Max Batch Size

```bash
python find_max_batch_size.py                        # mae_vit_small_patch32x8 on cuda:0
python find_max_batch_size.py --model mae_vit_ultra_light --device cuda:1
```

Binary-searches the largest batch that fits in VRAM using a full forward+backward+optimizer step.

## Architecture

```
mae/
├── model.py          # MaskedAutoencoderViT + FlashAttention + factory functions
├── dali_loader.py    # build_dali_loader — NVIDIA DALI pipeline backed by numpy memmap
├── dataset.py        # YiddishSharedInRamDataset (legacy, kept for reference)
└── pos_embed.py      # 2D sin-cos positional embeddings (supports rectangular grids)
train.py              # training entry point (torch.autocast bf16, DDP, gradient accumulation)
hpo.py                # hyperparameter search (Optuna + Lightning)
hpo_scripts/          # phase1_broad.sh, phase2_fine.sh, phase3_show_best.sh
find_max_batch_size.py # binary-search for max batch size
prepare_dataset.py    # one-time script: decode + resize images → ./data/yiddish_lines.npy
training_logger.py    # TrainingLogger — rich progress bar, VRAM tracking, profiler, summary
analyze_trace.py      # CLI tool: analyze PyTorch profiler .pt.trace.json files
analyze_nsys.py       # CLI tool: analyze Nsight Systems .nsys-rep / .sqlite files
analyze_ncu.py        # CLI tool: analyze Nsight Compute .ncu-rep files (per-kernel GPU metrics)
profile_nsys.sh       # wrapper: runs training under nsys, saves to nsys/<name>.nsys-rep
```

**MAE flow:** image -> patch embed -> random mask 75% -> encode visible patches only -> decoder re-inserts mask tokens -> reconstruct full image -> MSE loss on masked patches only.

**FlashAttention** wraps `F.scaled_dot_product_attention` and is injected into timm `Block` via `attn_layer=FlashAttention`.

**LinearPatchEmbed** — custom patch embedding using reshape + `nn.Linear` instead of timm's `PatchEmbed` (`nn.Conv2d`). For non-overlapping patches (stride == kernel_size) these are mathematically identical, but Conv2d triggers cuDNN workspace VMM (`cuMemSetAccess`/`cuMemUnmap`) that requires `cudaDeviceSynchronize` and causes ~33% GPU idle time. LinearPatchEmbed has no cuDNN dependency and is fully CUDA-graph-compatible.

**Factory functions** in `mae/model.py`: `mae_vit_base_patch16`, `mae_vit_large_patch16`, `mae_vit_huge_patch14`, `mae_vit_base_patch32x8_32x512`, `mae_vit_ultra_light`, `mae_vit_small_patch32x8` (current), `mae_vit_small_patch16x16`.

**Rectangular patch support** — `patch_size` accepts a `(ph, pw)` tuple throughout `MaskedAutoencoderViT` (`patchify`, `unpatchify`, `decoder_pred`, `_grid_size` all use `ph`/`pw` separately).

**Mixed precision** via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — runs matmuls in bf16, keeps numerically sensitive ops (softmax, layer norm, loss) in fp32 automatically.

## Key Hyperparameters

- **patch_size=(32, 8)** — full-height column patches; grid is (1, 64) for 32×512 images
- **mask_ratio=0.75** (75% patches masked, 25% visible to encoder) — consider 0.60–0.65 for 1D grids
- **norm_pix_loss=True** normalizes target patches to zero-mean unit-variance before MSE
- **per_gpu_batch=512**, **accum_steps=4** → effective batch = 512 × 4 × 2 GPUs = 4096
- **lr=1.5e-4 × (effective_batch / 256)** (linear scaling rule)
- **weight_decay=0.05**, **betas=(0.9, 0.95)** (MAE paper values; β2=0.95 speeds convergence)
- **clip_grad_norm_(max_norm=1.0)** — guards against gradient explosions

## Current Model — `mae_vit_small_patch32x8`

embed_dim=384, depth=6, num_heads=6, decoder_embed_dim=192, decoder_depth=4, patch_size=(32,8).
~12.7M parameters. Grid: (1, 64) = 64 patches.

## Gradient Accumulation

`train.py` supports `--accum-steps N` (default 4). Gradients accumulate over N micro-batches before one optimizer step. DDP all-reduce is suppressed on intermediate steps via `step_module.no_sync()`. LR scales with effective batch = per_gpu_batch × accum_steps × world_size.

## Data Pipeline

`prepare_dataset.py` runs once: decodes all images with PIL, resizes to 32×512, saves as
a single `(N, 32, 512)` uint8 numpy memmap (`./data/yiddish_lines.npy`). Decode and resize
cost is paid once, not every epoch.

`mae/dali_loader.py` — `build_dali_loader(npy_path, batch_size, num_threads, device_id)`:
- `np.load(..., mmap_mode='r')` memory-maps the file; OS manages page cache
- `fn.external_source(batch=True)` — callback called **once per batch** (not per sample); vectorized numpy fancy-index `data[idx]` fetches the whole batch in one call, avoiding GIL contention across 4 DALI threads
- `fn.crop_mirror_normalize` converts uint8 HWC → float32 CHW, normalizes to [0, 1] on GPU
- Returns a `DALIGenericIterator`; batches are already on GPU as `(N, 1, H, W)` float32
- Iterate: `batch = batch_data[0]["images"]` — no `.to()`, `.float()`, or `.div_()` needed

**Key DALI lesson:** `batch=False` calls Python once per sample — 5120 GIL acquisitions per batch at batch_size=5120. Always use `batch=True` with a vectorized source function.

## TrainingLogger

`TrainingLogger` in `training_logger.py` separates all visualization from training logic. Interface:

```python
with TrainingLogger(device, num_epochs, len(dataloader), profile) as logger:
    logger.begin_epoch(epoch)          # resets progress bar
    logger.on_step()                   # advances bar, tracks VRAM
    logger.end_epoch(epoch, avg_loss)  # syncs loss once, prints epoch summary

if profile:
    with logger.profile_step():        # captures one step → runs/<profile>_<timestamp>.pt.trace.json
        forward_backward(batch)
```

Loss is accumulated on GPU across steps (`epoch_loss += loss.detach()`) and synced to CPU
once per epoch via `.item()` — avoids per-step `cudaStreamSynchronize`.

Tracks: rich live progress bar, per-epoch summaries, peak VRAM, avg loss, steps/sec.

## Profiling

Two complementary tools:

**PyTorch profiler** (`--profile <name>`): captures one post-training step. Output in
`runs/<name>_<timestamp>.pt.trace.json`. Analyze with:
```bash
python analyze_trace.py runs/<name>_<timestamp>.pt.trace.json
```

**Nsight Systems** (`./profile_nsys.sh <name>`): captures the full training run. Output in
`nsys/<name>.nsys-rep`. Analyze with:
```bash
python analyze_nsys.py nsys/<name>.nsys-rep
```
`analyze_nsys.py` reads the SQLite export directly — sections: GPU device info, kernel summary
by type, GPU utilization/density, CUDA runtime API (syncs, launches, graph launches), NVTX
ranges (DALI pipeline stages), memory ops, bottleneck detection.

## Performance

Target: 2× RTX 3090 (24GB each). Optimizations:
- **bf16 autocast** — matmuls in bf16; loss/norm stays fp32
- **FlashAttention** via SDPA
- **LinearPatchEmbed** — reshape + `nn.Linear` instead of Conv2d; eliminates cuDNN workspace VMM and ~1300 DeviceSyncs per run (was causing ~33% GPU idle time)
- **DALI memmap pipeline** — no per-epoch decode/resize; `batch=True` source avoids GIL contention
- **Gradient accumulation** — `no_sync()` on intermediate steps suppresses unnecessary NCCL all-reduces
- **Loss sync once per epoch** — eliminates per-step `cudaStreamSynchronize`
- **Fused AdamW** (`fused=True`) — single `multi_tensor_apply_kernel` over all parameters
- **`clip_grad_norm_(max_norm=1.0)`** — guards against gradient explosions; loose threshold fires rarely during healthy training

## Profiling History (nsys runs)

| Run | Config | GPU Density | Notes |
|-----|--------|-------------|-------|
| 9   | ultra_light baseline | 98.5% | Reference — CUDA graphs, no Conv |
| TokenInteraction | +Conv1d depthwise | 70.2% | 48s DeviceSync; Conv1d removed |
| 17  | Conv1d removed, Conv2d (PatchEmbed) still present | 67.1% | 44s DeviceSync; PatchEmbed replaced |
| next| LinearPatchEmbed | ~98.5% expected | No cuDNN anywhere in model |

**Key lesson:** Any `nn.Conv*` in the training graph can trigger cuDNN workspace VMM
(`cuMemSetAccess`/`cuMemUnmap`), which requires `cudaDeviceSynchronize` and causes
seconds-scale GPU stalls. Use `nn.Linear` + reshape for all patch-embedding operations.
