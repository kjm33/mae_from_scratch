# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Masked Autoencoder (MAE) for self-supervised pretraining on Yiddish OCR text-line images. Grayscale 32x512px images, ViT-Base encoder with patch size 8. Based on *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2022).

## Training

```bash
# 2x GPU via accelerate (bf16 mixed precision, DDP)
bash run_train.sh

# Or directly:
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py
```

Training config is in `train.py` (no argparse). Dataset images go in `./data/yiddish_lines/`.

TensorBoard: `tensorboard --logdir ./runs/mae_yiddish`

## Architecture

```
mae/
├── model.py      # MaskedAutoencoderViT + FlashAttention + factory functions
├── dataset.py    # YiddishSharedInRamDataset (loads all images into shared memory)
└── pos_embed.py  # 2D sin-cos positional embeddings (supports rectangular grids)
train.py          # training entry point (accelerate + torch.compile)
```

**MAE flow:** image -> patch embed -> random mask 75% -> encode visible patches only -> decoder re-inserts mask tokens -> reconstruct full image -> MSE loss on masked patches only.

**FlashAttention** wraps `F.scaled_dot_product_attention` and is injected into timm `Block` via `attn_layer=FlashAttention`.

**Factory functions** in `mae/model.py`: `mae_vit_base_patch16`, `mae_vit_large_patch16`, `mae_vit_huge_patch14`, `mae_vit_base_patch8_32x512` (Yiddish config).

## Key Hyperparameters

- **mask_ratio=0.75** (75% patches masked, 25% visible to encoder)
- **norm_pix_loss=True** normalizes target patches to zero-mean unit-variance before MSE
- **Effective batch size** = `batch_size` x `num_gpus`. LR scales linearly: `lr = blr * effective_batch / 256`

## Performance

Target: 2x RTX 3090 (24GB each). Optimizations: bf16, torch.compile(mode="reduce-overhead"), FlashAttention via SDPA, in-RAM dataset with shared memory, persistent workers.
