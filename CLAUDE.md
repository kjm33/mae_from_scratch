# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Masked Autoencoder (MAE) for self-supervised pretraining on Yiddish OCR text-line images. Grayscale 32x512px images, ViT-Base encoder with patch size (32h, 8w). Based on *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2022).

## Training

```bash
# Prepare dataset once (decode + resize all images into a single memmap file)
python prepare_dataset.py

# Single GPU
python train.py

# With PyTorch profiler — captures one post-training step to runs/<name>_<timestamp>.pt.trace.json
python train.py --profile <name>

# With Nsight Systems profiler — captures full run to nsys/<name>.nsys-rep
./profile_nsys.sh <name>
```

Training config is in `train.py`. Dataset images go in `./data/yiddish_lines/`.
The prepared memmap is written to `./data/yiddish_lines.npy`.

TensorBoard: `tensorboard --logdir ./runs/mae_yiddish`

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
analyze_nsys.py       # CLI tool: analyze Nsight Systems .nsys-rep / .sqlite files
analyze_ncu.py        # CLI tool: analyze Nsight Compute .ncu-rep files (per-kernel GPU metrics)
profile_nsys.sh       # wrapper: runs training under nsys, saves to nsys/<name>.nsys-rep
```

**MAE flow:** image -> patch embed -> random mask 75% -> encode visible patches only -> decoder re-inserts mask tokens -> reconstruct full image -> MSE loss on masked patches only.

**FlashAttention** wraps `F.scaled_dot_product_attention` and is injected into timm `Block` via `attn_layer=FlashAttention`.

**LinearPatchEmbed** — custom patch embedding using reshape + `nn.Linear` instead of timm's `PatchEmbed` (`nn.Conv2d`). For non-overlapping patches (stride == kernel_size) these are mathematically identical, but Conv2d triggers cuDNN workspace VMM (`cuMemSetAccess`/`cuMemUnmap`) that requires `cudaDeviceSynchronize` and causes ~33% GPU idle time. LinearPatchEmbed has no cuDNN dependency and is fully CUDA-graph-compatible.

**Factory functions** in `mae/model.py`: `mae_vit_base_patch16`, `mae_vit_large_patch16`, `mae_vit_huge_patch14`, `mae_vit_base_patch32x8_32x512`, `mae_vit_ultra_light` (current Yiddish config).

**Rectangular patch support** — `patch_size` accepts a `(ph, pw)` tuple throughout `MaskedAutoencoderViT` (`patchify`, `unpatchify`, `decoder_pred`, `_grid_size` all use `ph`/`pw` separately).

**Mixed precision** via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — runs matmuls in bf16, keeps numerically sensitive ops (softmax, layer norm, loss) in fp32 automatically.

## Key Hyperparameters

- **patch_size=(32, 8)** — full-height column patches; grid is (1, 64) for 32×512 images
- **mask_ratio=0.75** (75% patches masked, 25% visible to encoder)
- **norm_pix_loss=True** normalizes target patches to zero-mean unit-variance before MSE
- **batch_size=9216** (1024×9), **lr=4.8e-3** (linear scaling from base 1.5e-4 at batch 256), **weight_decay=0.05**

## Current Model — `mae_vit_ultra_light`

embed_dim=256, depth=6, num_heads=4, decoder_embed_dim=128, decoder_depth=2, patch_size=(32,8).
5,161,344 parameters. ~10ms GPU per step.

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
with TrainingLogger(device, num_epochs, len(dataloader), profile) as logger:
    logger.begin_epoch(epoch)          # resets progress bar
    logger.on_step()                   # advances bar, tracks VRAM
    logger.end_epoch(epoch, avg_loss)  # syncs loss once, prints epoch summary

if profile:
    with logger.profile_step():        # captures one step → runs/<profile>_<timestamp>.pt.trace.json
        train_step(batch)
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

Target: RTX 3090 (24GB). Optimizations:
- **bf16 autocast** — matmuls in bf16; loss/norm stays fp32
- **`torch.compile(mode="max-autotune")`** — benchmarks kernel configs on first run, caches to `.cache/`; subsequent runs load from cache
- **Full train_step compiled** — `zero_grad` + forward + backward + optimizer captured into two CUDA graphs; CPU only fires two `cudaGraphLaunch` calls per step (~1ms total CPU overhead)
- **`torch.profiler.record_function`** annotations inside `train_step` — preserved by `torch.compile`, visible in TensorBoard trace; zero overhead when profiler is inactive
- **FlashAttention** via SDPA
- **LinearPatchEmbed** — reshape + `nn.Linear` instead of Conv2d; eliminates cuDNN workspace VMM and ~1300 DeviceSyncs per run (was causing ~33% GPU idle time)
- **DALI memmap pipeline** — no per-epoch decode/resize
- **Loss sync once per epoch** — eliminates per-step `cudaStreamSynchronize`
- **Fused AdamW** (`fused=True`) — single `multi_tensor_apply_kernel` over all parameters
- **No `clip_grad_norm_`** — removed; was causing a blocking sync + ~0.7 ms kernel overhead per step

`mae_vit_ultra_light` steady-state: ~10ms GPU per step, ~98.5% kernel density (single-step trace),
two CUDA graph launches. Triton fused kernels dominate, FlashAttention backward ~15%, GEMM ~25%.

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
