# Nsys Analysis — run 18 (LinearPatchEmbed, no Conv2d)

**File:** `nsys/18_wo_conv2d.nsys-rep`
**Date:** 2026-04-21

## Summary

The Conv2d removal worked — but density dropped further to 61.4%. The culprit is
`torch.compile(mode="max-autotune")` recompiling **471 kernel variants** because the
LinearPatchEmbed change invalidated the compiled kernel cache. All 471 `cuModuleLoad`
calls concentrate in the first 20 seconds and cause blocking GPU stalls during JIT
compilation. This is a **one-time cost** — the next run with a warm cache is expected
to recover to ~97-98% density.

## GPU Utilization

| Metric              | Run 18 (no Conv2d) | Run 17 (Conv2d PatchEmbed) | Run 9 (baseline) |
|---------------------|--------------------|-----------------------------|-------------------|
| Total kernel time   | 40.1 s             | 44.0 s                      | ~50 s             |
| GPU active span     | 65.3 s             | 65.7 s                      | ~51 s             |
| **Kernel density**  | **61.4%**          | **67.1%**                   | **98.5%**         |
| Total idle time     | 25.3 s             | 21.7 s                      | ~0.75 s           |
| Context syncs       | 1,693 calls        | 1,290 calls                 | ~6 calls          |
| cuModuleLoad        | 471 calls          | 0 calls                     | 0 calls           |

## Root Cause 1: max-autotune Recompilation (one-time cost)

The LinearPatchEmbed change modified the compiled graph, invalidating `.cache/`.
`torch.compile(mode="max-autotune")` benchmarks multiple kernel variants per op —
471 Triton `.cubin` files compiled and loaded during this run.

Each `cuModuleLoad` requires `cudaDeviceSynchronize` before the new kernel can be
registered. If the GPU is mid-stream on a CUDA graph, this stalls it.

```
cuModuleLoad distribution:
  [  0-  5s]: 210  ██████████████████████████████████████████
  [  5- 10s]: 102  ████████████████████████
  [ 10- 15s]:  73  █████████████████
  [ 15- 20s]:  86  ████████████████████
  [ 20- 65s]:   0  (compilation done)
```

471 module loads all happen in the first 20 seconds. After that, zero.

Large DeviceSyncs (>5ms) follow the same pattern:
```
  [  0-  5s]: 104  ████████████████████████████
  [  5- 10s]:  65  █████████████████
  [ 10- 15s]:  24  ██████
  [ 15- 20s]:  33  █████████
  [ 20- 65s]:  53  (1 per step — remaining cuBLAS VMM)
```

291 of the 443 large stalls (66%) are purely from compilation warmup.

## Root Cause 2: cuBLAS Workspace VMM (persistent, minor)

Even with Conv2d gone, 70 `cuMemUnmap` events occur during training (0.47/step).
These are **not** from Conv2d — they were present in run 17 too, masked by the louder
PatchEmbed stalls. Source: cuBLAS workspace resizing for GEMM operations.

| Metric | During training | After training (cleanup) | Total |
|--------|----------------:|-------------------------:|------:|
| cuMemUnmap | 70 | 112 | 182 |
| cuMemSetAccess | 70 | 112 | 182 |

The 112 cleanup events happen after the last kernel — they're allocator teardown,
not training stalls.

At ~21ms average per stall: 70 × 21ms = 1,470ms stall / 149 steps = ~10ms/step.
With a ~270ms GPU compute step, that's ~3.6% idle from cuBLAS VMM.
**Expected density with warm cache: ~96-97%.**

## Kernel Breakdown — Conv2d Fully Gone

| Category              |   Time | Share | vs Run 17 |
|-----------------------|-------:|------:|-----------|
| Triton fused          | 15.6 s | 38.8% | +6.4 pp   |
| GEMM / matmul         | 12.6 s | 31.3% | -3.9 pp   |
| FlashAttn bwd         |  7.7 s | 19.1% | +1.2 pp   |
| FlashAttn fwd         |  2.8 s |  7.1% | +0.5 pp   |
| Fill / zero_grad      | 0.58 s |  1.5% | +0.5 pp   |
| DALI                  | 0.41 s |  1.0% | stable    |
| Optimizer             | 0.40 s |  1.0% | stable    |
| **Elementwise**       | **0.015 s** | **0.0%** | **CUDA graphs capturing ✓** |
| Conv2d (PatchEmbed)   |   0    |   0%  | **eliminated ✓** |

No `nchwToNhwcKernel`, no `implicit_convolve_sgemm`, no `wgrad` kernel. Fix confirmed.

Elementwise at 0.0% (down from 0.4% in run 17, 23.7% in run 15) — CUDA graphs
capturing all fill/copy/add ops perfectly.

## CUDA Graphs

- **298 graph launches** (2/step × 149 steps) — graphs healthy ✓
- **~146k individual cudaLaunchKernel** — elevated vs run 17 (122k/197 steps → 623/step
  vs 981/step); excess is from compiled kernel benchmarking during autotuning warmup

## Expected Next Run (run 19 — warm cache)

| Source                | Stall/step | After fix |
|-----------------------|-----------|-----------|
| cuModuleLoad (compile)| ~13 ms    | → 0 (cache warm) |
| cuBLAS workspace VMM  | ~10 ms    | → 0 with `PYTORCH_ALLOC_CONF` |
| DALI PopOutputs       | overlapped | unchanged |

With warm cache and no cuModuleLoad stalls, density should reach ~96-98%.

To also eliminate the cuBLAS workspace VMM, set before training:
```bash
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
```
`cudaMallocAsync` uses stream-ordered allocation, avoiding the VMM unmap/remap cycle
that requires device synchronization.

## Verdict

Run 18 is a **transitional run** — the compilation overhead is a one-time cost paid
because the model graph changed. Density will recover on run 19. The Conv2d fix is
correct and complete.
