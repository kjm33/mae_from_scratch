# EfficientViT — What's Applicable to This Project

**Paper:** EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention (Liu et al., 2023)

**Context:** evaluated against mae_vit_ultra_slim — 64 tokens, 1D sequence, MAE pretraining,
torch.compile + CUDA graphs, 98.5% GPU density, ~10ms/step.

---

## 1. Sandwich Layout — Not applicable

**What it does:** replaces the 1:1 MHSA:FFN ratio with N FFN layers surrounding one MHSA.
Motivation: tensor reshaping and element-wise ops in MHSA are memory-bound and dominate
runtime (44.2% of Swin-T on their profiler).

**Why it doesn't apply:** sequence is 64 tokens. MHSA is O(L²·D) — at L=64 it's tiny.
The memory-bandwidth problem EfficientViT solves only manifests at hundreds of tokens with
large feature maps. With torch.compile + Triton fusion, the reshape overhead they measured is
already eliminated.

---

## 2. Cascaded Group Attention (CGA) — Marginal

**What it does:** splits the full feature across heads (head j sees its chunk of the
embedding), then cascades: head j's input = its feature split + head (j−1)'s output. Forces
diversity between heads, reduces redundant attention maps (Fig. 4 in paper shows heads in the
same layer converge to nearly identical maps).

**For this project:** the diversity argument is real even for 4 heads, but:
- For pretraining rather than supervised inference, the benefit is less established
- With only 4 heads and 64 tokens, redundancy is a smaller problem than in 12-head models
- Implementation requires rewriting the attention module (breaks the current timm.Block +
  FlashAttention injection pattern)

**Verdict:** potentially improves representation diversity but adds real implementation
complexity. Not a clear win for MAE pretraining.

---

## 3. Token Interaction (DWConv) — Worth trying

**What it does:** adds a depthwise convolution before each FFN to give each token local
awareness of its neighbors. Costs almost nothing (one depthwise conv per block = 1 op per
channel, no cross-channel mixing).

**Why it fits:** the 64 tokens are horizontal character-column strips. Adjacent tokens are
neighboring column slices of the same word — there is genuine local structure. A 1D DWConv
(kernel=3) on the token sequence lets each token incorporate its two neighbors before the FFN
processes it. Standard ViT attention is global only; this technique addresses local text
structure directly and cheaply.

Implementation — add before each FFN in encoder blocks:
```python
nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
```
torch.compile will fuse it. Adds ~`2 × depth × embed_dim × L` FLOPs — negligible.

---

## Summary

| Technique              | Worth applying | Reason |
|------------------------|----------------|--------|
| Sandwich Layout        | No             | Memory-bound MHSA problem doesn't exist at 64 tokens |
| Cascaded Group Attention | Maybe        | Diversity benefit exists; complex to implement; unclear gain for MAE pretraining |
| DWConv Token Interaction | **Yes**      | Cheap, injects local character-context the global attention misses |
| Q,K dimension reduction | No            | FLOPs savings negligible at embed_dim=256, L=64 |
| ReLU over GELU         | No             | GELU better for pretraining quality; no speedup with compile |

**Bottom line:** EfficientViT is designed for inference-speed-constrained supervised
classification on ImageNet (large feature maps, many tokens, mobile deployment). Most of its
wins solve problems this project doesn't have. The DWConv token interaction is the one clean
transfer — cheap and directly addresses local structure of text lines.
