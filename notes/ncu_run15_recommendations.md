# Recommendations from NCU Run 15 Analysis

Based on `ncu/15_large_batch_size.ncu-rep` — see `ncu_run15_large_batch_analysis.md` for full context.

## Priority 1: Benchmark CUDA graphs back on vs. current setup

Elementwise kernels (2,964 invocations) eat 23.7% of GPU time purely from launch overhead —
exactly what CUDA graphs eliminate. Tradeoff from commit `073cb68` was batch 5×1024 → 8×1024
by disabling graphs.

Unknown: which actually wins on **samples/sec**. Run a timed comparison:
- Option A (current): `compile(default)`, batch=8192
- Option B: `compile(reduce-overhead)`, batch=5120

Measure wall-clock samples/sec for a few hundred steps. The 60% larger batch of option A
may not compensate for the extra ~30ms of elementwise overhead per step.

## Priority 2: Remove `clip_grad_norm_`

```python
# train.py line 124 — remove this:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Generates several small-SM kernels (`GradNorm` 0.27ms, `MultiTensor` 0.45ms, multiple
reduces at 9% SM) on every step. With `OneCycleLR` and a tiny model, gradient explosion
is extremely unlikely — the MAE paper doesn't use grad clipping. Cheapest win.

## Priority 3: Switch to `max-autotune` (already in the code, just commented out)

```python
# train.py lines 109-110:
@torch.compile(mode="max-autotune")  # uncomment
# @torch.compile(mode="default")    # comment out
```

GEMMs at 30.6% SM because `default` mode doesn't benchmark tile configs. `max-autotune`
picks the best tiling for the exact matrix shapes (embed_dim=256, ~16 visible patches ×
8192 batch) and caches to `.cache/`. One-time compile cost, faster every subsequent run.

## What's Not Worth Chasing

- **GEMM 8.33% occupancy**: register-bound (222 regs/thread), fundamental to Ampere bf16
  tensor-core kernels at embed_dim=256. No config change fixes it.
- **LayerNorm bwd dγ/dβ at 4.5% SM**: inherent cross-batch reduction; needs custom kernel.
