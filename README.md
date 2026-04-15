# MAE from Scratch

A from-scratch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2022), applied to **self-supervised pretraining on Yiddish OCR text-line images**.

The encoder learns rich representations by reconstructing randomly masked patches — no labels needed. The pretrained encoder can then be fine-tuned for downstream tasks like OCR or text-line classification.

## How MAE Works

1. An input image is split into non-overlapping patches
2. 75% of patches are randomly masked (discarded)
3. Only the **visible 25%** are fed through a ViT encoder — this makes training efficient
4. A lightweight decoder receives encoder output + learnable mask tokens and reconstructs the full image
5. The loss is MSE computed **only on masked patches** — forcing the model to learn meaningful features, not just copy visible pixels

## Architecture

The model is a Vision Transformer (ViT) adapted for **rectangular grayscale images** (32x512 px). Patches are **full-height columns*** of size `(32, 8)`, producing a `(1, 64)` grid — 64 patches per image.

Current config (`mae_vit_ultra_slim`):

| Component         | Value    |
| ----------------- | -------- |
| Encoder embed dim | 256      |
| Encoder depth     | 6 layers |
| Encoder heads     | 4        |
| Decoder embed dim | 128      |
| Decoder depth     | 2 layers |
| Patch size        | (32, 8)  |
| GPU step time     | ~10 ms   |

## Project Structure

```
mae/
  model.py            # MaskedAutoencoderViT + FlashAttention + model factory functions
  dali_loader.py      # NVIDIA DALI data pipeline backed by numpy memmap
  dataset.py          # Legacy PyTorch Dataset (kept for reference)
  pos_embed.py        # 2D sin-cos positional embeddings (rectangular grid support)
train.py              # Training entry point (torch.compile + bf16 autocast)
prepare_dataset.py    # One-time: decode + resize images -> single memmap .npy file
training_logger.py    # Rich progress bar, VRAM tracking, profiler integration
analyze_trace.py      # Analyze PyTorch profiler traces
analyze_nsys.py       # Analyze Nsight Systems profiles
profile_nsys.sh       # Run training under Nsight Systems
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3090)
- [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html) (installed separately)

### Install

```bash
pip install -r requirements.txt
```

### Prepare Data

Place your images in `./data/yiddish_lines/`, then build the memmap file:

```bash
python prepare_dataset.py
```

This decodes and resizes all images to 32x512 grayscale, saving them as a single `(N, 32, 512)` uint8 numpy memmap at `./data/yiddish_lines.npy`. This cost is paid once — not every epoch.

### Train

```bash
python train.py
```

Monitor with TensorBoard:

```bash
tensorboard --logdir ./runs/mae_yiddish
```

## Profiling

Two complementary profiling tools are included:

**PyTorch profiler** — captures a single post-training step:

```bash
python train.py --profile <name>
python analyze_trace.py runs/<name>_<timestamp>.pt.trace.json
```

**Nsight Systems** — captures the full training run:

```bash
./profile_nsys.sh <name>
python analyze_nsys.py nsys/<name>.nsys-rep
```

## Optimization Journey

The training pipeline evolved through a series of profiler-driven optimizations. Each step was motivated by profiling data (PyTorch profiler traces and Nsight Systems captures), not guesswork. All measurements below use 7,279 images, batch_size=256, 6 epochs on an RTX 3090.

### Step 0 — Baseline

Starting point: ViT-Base MAE (patch 8x8, 256 tokens per image), HuggingFace Accelerate wrapper, PyTorch DataLoader with in-RAM dataset, `torch.compile(mode="reduce-overhead")`, bf16 autocast, bitsandbytes AdamW8bit.

| Metric             | Value   |
| ------------------ | ------- |
| Epoch time         | ~12.5 s |
| Peak VRAM          | 16.6 GB |
| GPU kernel density | —       |

### Step 1 — Remove Accelerate

Profiling revealed that the HuggingFace Accelerate wrapper added unnecessary abstraction for a single-GPU setup. Replacing it with direct `model.to(device)` + manual autocast simplified the code and eliminated hidden overhead.

| Metric             | Before | After                                     |
| ------------------ | ------ | ----------------------------------------- |
| First epoch        | 17.2 s | 79.7 s (torch.compile warmup now visible) |
| Steady-state epoch | 12.5 s | 12.4 s                                    |
| Steps/sec          | 2.24   | 1.23 (amortized over warmup)              |

**Insight:** Accelerate was hiding `torch.compile` warmup cost by compiling lazily. Without the wrapper, the first-epoch compilation penalty became explicit — but steady-state was identical, and the code was now transparent and debuggable.

### Step 2 — 8-bit Optimizer

Switched to `bitsandbytes.optim.AdamW8bit` to reduce optimizer state memory. Saved ~600 MB VRAM.

| Metric     | Before  | After   |
| ---------- | ------- | ------- |
| Peak VRAM  | 16.6 GB | 16.0 GB |
| Epoch time | 12.4 s  | 12.6 s  |

**Trade-off:** Marginal VRAM win but introduced a hidden cost discovered later (Step 5).

### Step 3 — DALI Memmap Data Pipeline

Replaced the PyTorch `DataLoader` + in-RAM `Dataset` with an NVIDIA DALI pipeline backed by a numpy memmap file. A one-time `prepare_dataset.py` script decodes and resizes all images into a single `(N, 32, 512)` uint8 `.npy` file. DALI reads it via memory-mapping, normalizes on GPU, and delivers batches already on device as float32.

| Metric      | Before               | After         |
| ----------- | -------------------- | ------------- |
| Epoch time  | 12.6 s               | 11.8 s        |
| Data on GPU | needed `.to(device)` | already there |

**Why it matters:** Decode and resize cost was paid every epoch with the old loader. Now it's paid once (in `prepare_dataset.py`), and the OS page cache handles the rest. No `.to()`, `.float()`, or `.div_()` calls needed in the training loop.

### Step 4 — Async Loss Accumulation

Profiling (run 5) revealed that `loss.item()` called every step was triggering `cudaStreamSynchronize`, blocking the CPU for **187–403 ms per step** until the GPU finished. This completely negated the benefit of CUDA graphs.

**Fix:** Accumulate loss as a detached GPU tensor across all steps, call `.item()` once at epoch end.

| Metric            | Before          | After         |
| ----------------- | --------------- | ------------- |
| CPU sync per step | 187–403 ms      | 0 ms          |
| Syncs per epoch   | 28 (every step) | 1 (epoch end) |

### Step 5 — Replace 8-bit Optimizer with Fused AdamW

Profiling revealed that `bitsandbytes.AdamW8bit` issued **252 `cudaDeviceSynchronize` calls per step** — one per parameter update. Each sync was cheap (~16 us), but they serialized all optimizer work and prevented kernel pipelining. The optimizer phase took 24.8 ms wall time for only 6 ms of actual GPU compute — **18.8 ms of pure sync overhead**.

**Fix:** Switched to `torch.optim.AdamW(fused=True)` — a single `multi_tensor_apply_kernel` over all parameters, zero per-parameter syncs.

| Metric               | Before (AdamW8bit) | After (fused AdamW) |
| -------------------- | ------------------ | ------------------- |
| Optimizer syncs/step | 252                | 0                   |
| Optimizer wall time  | 24.8 ms            | ~6 ms               |
| VRAM                 | 16.0 GB            | 16.6 GB (+600 MB)   |

**Trade-off:** Used ~600 MB more VRAM (fp32 optimizer states vs 8-bit), but eliminated the sync bottleneck entirely. Worth it on a 24 GB card.

### Step 6 — Compile Full train_step + max-autotune

Extended `torch.compile` to cover the **entire `train_step`** function — `zero_grad` + forward + backward + optimizer — and switched from `mode="reduce-overhead"` to `mode="max-autotune"`.

`max-autotune` benchmarks multiple GEMM tilings and Triton kernel configurations on the first run, caching results to `.cache/`. Subsequent runs load from cache with no penalty.

The result: the entire training step is captured into **two CUDA graphs**. The CPU fires two `cudaGraphLaunch` calls per step (~1 ms total CPU overhead).

| Metric                   | Before           | After                 |
| ------------------------ | ---------------- | --------------------- |
| GPU kernel density       | 90.7%            | 99.7%                 |
| CUDA graph launches/step | 2 (fwd+bwd only) | 2 (fwd+bwd+optimizer) |
| CPU overhead/step        | ~25 ms           | ~1 ms                 |

### Step 7 — Rectangular Patches (32x8)

The original 8x8 patches produced 256 tokens per 32x512 image — far more than necessary for narrow text lines. Switching to full-height `(32, 8)` patches reduced the token count to 64, cutting attention cost quadratically.

| Metric          | Before (8x8, 256 tokens) | After (32x8, 64 tokens) |
| --------------- | ------------------------ | ----------------------- |
| Epoch time      | ~12 s                    | ~3.4 s                  |
| Peak VRAM       | 16.6 GB                  | 5.4 GB                  |
| Step time (GPU) | ~197 ms                  | ~35 ms                  |

**3.5x faster, 3x less VRAM.** The biggest single optimization. Full-height patches make physical sense for text lines — each patch is a character-width column spanning the full line height.

### Step 8 — Ultra-Slim Model

The original ViT-Base (embed_dim=768, depth=12, heads=12) was designed for ImageNet classification of 256x256 RGB images — massive overkill for 32x512 grayscale text lines. Replaced with `mae_vit_ultra_slim`: embed_dim=256, depth=6, heads=4, decoder_embed=128, decoder_depth=2.

| Metric        | Before (ViT-Base, 32x8) | After (ultra_slim, 32x8) |
| ------------- | ----------------------- | ------------------------ |
| Epoch time    | 3.4 s                   | 0.3 s                    |
| First epoch   | 230 s                   | 12.8 s                   |
| Peak VRAM     | 5.4 GB                  | 0.5 GB                   |
| Steps/sec     | 0.68                    | 11.73                    |
| GPU step time | ~35 ms                  | ~10 ms                   |

**11x faster epochs, 10x less VRAM.** The model is now right-sized for the task.

### Final State

| Metric                   | Baseline                       | Final                      |
| ------------------------ | ------------------------------ | -------------------------- |
| Epoch time               | 12.5 s                         | 0.3 s                      |
| GPU step time            | ~222 ms                        | ~10 ms                     |
| Peak VRAM                | 16.6 GB                        | 0.5 GB                     |
| GPU kernel density       | 90.7%                          | 98.5%                      |
| CPU overhead/step        | ~227 ms                        | ~1 ms                      |
| CUDA graph launches/step | 0                              | 2                          |
| Data pipeline            | PyTorch DataLoader             | DALI memmap (GPU-resident) |
| Optimizer                | bnb AdamW8bit (252 syncs/step) | Fused AdamW (0 syncs)      |
| Loss sync                | Every step (cudaStreamSync)    | Once per epoch             |

Two CUDA graph launches per step. Triton fused kernels dominate (44%), FlashAttention backward 15%, optimizer 10%. The GPU is compute-bound in a healthy way — no idle gaps, no sync stalls, no data loading bottlenecks.

## Key Hyperparameters

| Parameter             | Value  |
| --------------------- | ------ |
| Mask ratio            | 0.75   |
| Batch size            | 256    |
| Learning rate         | 1.5e-4 |
| Weight decay          | 0.05   |
| Normalized pixel loss | Yes    |



## Plan

### Add convolutional layers

During reading [Transformers for Natural Language Processing and Computer Vision - Third Edition [Book]](https://www.oreilly.com/library/view/transformers-for-natural/9781805128724/) I noticed that decribed ViT model (Google/ViT-Base-Patch16–224) contains CNNs.



### Changing positional encoding to RoPE

### Using only x-axis position encoding

### Research and apply optimization methods from EfficientViT

EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention

## References

- He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- [Facebook Research MAE](https://github.com/facebookresearch/mae) — original implementation

## License

[MIT](LICENSE)
