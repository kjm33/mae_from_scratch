# NCU Analysis — run 15 (large batch size)

**File:** `ncu/15_large_batch_size.ncu-rep`
**Date:** 2026-04-20

125.3 ms total GPU time, 4,999 kernel invocations.
*(NCU adds per-kernel serialization overhead; real step time is ~10 ms)*

## Time Breakdown by Category

| Category        |  Time | Share | Avg SM% | Notes |
|-----------------|------:|------:|--------:|-------|
| GEMM            | 31.9 ms | 25.4% | 30.6% | Small embed_dim=256/128 → tiles don't fill 82 SMs |
| Elementwise     | 29.7 ms | 23.7% | 11.4% | **2,964 invocations** — fill, copy, add scattered across step |
| LayerNorm       | 14.2 ms | 11.4% | 40.5% | Bwd dγ/dβ kernel at 4.5% SM (cross-batch reduction) |
| FlashAttn bwd   | 13.1 ms | 10.5% | 21.2% | |
| FusedAdamW      | 11.6 ms |  9.3% | 23.3% | 6 invocations at 7.7% SM — bias/small-tensor param groups |
| Reduce          |  6.3 ms |  5.0% |  9.1% | Loss, softmax numerics — very small reductions |
| FlashAttn fwd   |  5.1 ms |  4.1% | 36.3% | |
| GELU            |  3.3 ms |  2.6% | 44.2% | |
| Transpose       |  2.3 ms |  1.9% | 59.4% | |
| Scatter/Gather  |  1.2 ms |  1.0% | 12.6% | Mask indexing |
| GEMM split-K    |  0.7 ms |  0.6% | 16.6% | |

## Occupancy Summary

- Weighted-avg SM throughput: **28.6%**
- **50% of all kernel invocations have SM < 10%** — model is too small for 82 SMs

## Key Findings

### 1. Elementwise fragmentation (23.7%, 2,964 invocations)
Each fill/copy/add launches a tiny kernel at 11.4% avg SM utilization. Accumulates from
`zero_grad()`, dtype casts, and non-fused ops. With CUDA graphs disabled (traded for
larger batch size in run 13), every one fires a real kernel launch.

### 2. GEMM occupancy is register-bound at 8.33%
All GEMM kernels use 222 registers/thread → hard-caps occupancy at 4 warps/SM (8.33%).
Intentional: Ampere bf16 tensor-core kernels trade occupancy for deep software pipelining
(stages_32x5 etc.). Not a bug — just the ceiling for embed_dim=256.

### 3. LayerNorm bwd dγ/dβ at 4.5% SM (12 invocations, ~1.3 ms)
Cross-batch reduction for learnable scale/bias is inherently sequential at small sequence
lengths. Hard to fix without a custom fused kernel.

### 4. FusedAdamW bimodal SM%
~7.7% SM for small param groups (biases/embedding rows), ~28% for weight matrices.
Low-SM invocations are unavoidable given the model's small tensors.

### 5. Model is too small for the GPU
The ultra-slim config (embed_dim=256, depth=6) cannot fill 82 SMs at any batch size.
Batch scaling helped but embed_dim is the fundamental ceiling on parallelism.

## Most Actionable Next Step

Re-enabling CUDA graphs (`torch.compile` full step) would fuse/eliminate most of those
2,964 elementwise kernels into two `cudaGraphLaunch` calls, recovering ~30 ms. This was
previously traded off to allow larger batch size (commit 073cb68).

## Tooling

Analysis done with `analyze_ncu.py` (wraps `ncu --import ... --csv --metrics ...`):
```bash
python analyze_ncu.py ncu/15_large_batch_size.ncu-rep
```
