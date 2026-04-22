# Hyperparameter Tuning Notes

Observed during mae_vit_small_patch32x8 training (April 2026).

## 1. LR Schedule — replace OneCycleLR with cosine + linear warmup

OneCycleLR's aggressive ramp caused a gradient explosion spike at epochs 8–9 (loss jumped
from 0.534 back to 0.703, took 12 epochs to recover). Standard MAE uses cosine decay with
linear warmup — more stable over hundreds of epochs.

## 2. AdamW β2: 0.999 → 0.95

MAE paper uses `betas=(0.9, 0.95)`. Lower β2 forgets stale gradients faster — noticeably
speeds up convergence.

## 3. More epochs

50 epochs (11,600 steps) is too few — loss was still declining at epoch 31. 200–300 epochs
is a better target given ~238k training images.

## 4. Mask ratio: try 0.6–0.65 instead of 0.75

Patch grid is 1×64. At 75% masking the encoder sees only 16 patches — very sparse for a
text line. Lower ratio gives easier task, faster convergence, and more encoder context for
downstream OCR.

## Priority

Apply β2=0.95 and cosine schedule first (low risk, high reward), then experiment with mask
ratio, then commit to a long run (200+ epochs).
