# CNN Stem — Should We Add It?

**Context:** Considered adding CNN layers "as in vit-base-patch16-224" to the MAE encoder.

## What that actually means

The pure `vit_base_patch16_224` has no CNN layers — `PatchEmbed` is a single
`Conv2d(in_chans, embed_dim, kernel=16, stride=16)`. Our model already does the same.

The "hybrid ViT" (e.g. timm's `vit_base_r50_s16_224`) uses a ResNet stem as a feature
extractor before the transformer. That's the actual CNN-augmented variant.

## Why it makes less sense for this setup

1. **Patch is already full-height** — `(32,8)` captures the entire vertical extent of a
   character column in one shot. A CNN stem's value is building hierarchical spatial features
   bottom-up (edges → shapes → objects). With a `(1,64)` grid the sequence is 1D; there's no
   vertical hierarchy to exploit.

2. **MAE task vs. recognition** — CNN stems help most in supervised settings at scale (Xiao
   et al. 2021, *Early Convolutions Help Transformers See Better*, showed stability benefits
   for ImageNet classification). For reconstruction-based self-supervised learning, plain
   `PatchEmbed` works extremely well and is what the original MAE paper uses deliberately.

3. **Diacritics (nekudes) concern** — small marks could be missed by an 8px-wide patch, but
   this is better addressed by halving patch width to `(32,4)` → 128 tokens, which is cheaper
   and preserves the current architecture cleanly.

## Better alternatives if richer features are needed

| Option | Effect | Cost |
|---|---|---|
| `patch_size=(32,4)` | 2× token count, finer horizontal resolution | no arch change |
| `depth=8` or `depth=10` | more capacity for long-range line context | linear in depth |
| `embed_dim=384` | more capacity per token | quadratic in attention |

## When a CNN stem *would* make sense

If fine-tuning for a downstream **recognition task** (CTC / attention decoder), a CNN stem
gives a useful inductive bias for multi-scale stroke features. For MAE pretraining on 32×512
text lines, the added complexity doesn't justify itself.
