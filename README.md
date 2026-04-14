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

The model is a Vision Transformer (ViT) adapted for **rectangular grayscale images** (32x512 px). Patches are full-height columns of size `(32, 8)`, producing a `(1, 64)` grid — 64 patches per image.

Current config (`mae_vit_ultra_slim`):

| Component | Value |
|---|---|
| Encoder embed dim | 256 |
| Encoder depth | 6 layers |
| Encoder heads | 4 |
| Decoder embed dim | 128 |
| Decoder depth | 2 layers |
| Patch size | (32, 8) |
| GPU step time | ~10 ms |

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

## Performance Optimizations

- **bf16 mixed precision** — matmuls/convs in bfloat16, numerically sensitive ops stay in fp32
- **`torch.compile(mode="max-autotune")`** — full `train_step` compiled into CUDA graphs; CPU fires just two `cudaGraphLaunch` calls per step (~1 ms CPU overhead)
- **FlashAttention** via `F.scaled_dot_product_attention`
- **DALI memmap pipeline** — GPU-side normalization, zero per-epoch decode cost
- **Fused AdamW** — single multi-tensor kernel over all parameters
- **Loss sync once per epoch** — no per-step `cudaStreamSynchronize`

Steady-state: ~10 ms/step, 98.5% GPU kernel density, Triton fused kernels 44%, FlashAttention backward 15%, optimizer 10%.

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Mask ratio | 0.75 |
| Batch size | 256 |
| Learning rate | 1.5e-4 |
| Weight decay | 0.05 |
| Normalized pixel loss | Yes |

## References

- He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- [Facebook Research MAE](https://github.com/facebookresearch/mae) — original implementation

## License

[MIT](LICENSE)
