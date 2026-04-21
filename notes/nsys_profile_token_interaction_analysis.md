# Nsys Analysis — `--profile` run (TokenInteraction model)

**File:** `nsys/--profile.nsys-rep`
**Date:** 2026-04-21
**Note:** Run name `--profile` — `./profile_nsys.sh --profile` passed the flag as the name.

## Model Change vs Prior Runs

`mae_vit_ultra_light` with `TokenInteractionBlock` — each of 6 encoder blocks now has
a depthwise Conv1d inserted before the MLP:

```
norm_ti → Conv1d(dim, dim, k=3, groups=dim) → residual → norm → MLP
```

Visible in kernel table: `conv_depthwise2d_forward`, `conv_depthwise2d_backward`,
`nchwToNhwc`, `wgrad` kernels — ~5.2 s GPU time (~10% of total).

## Critical Problem: cudaDeviceSynchronize

```
[HIGH] cudaDeviceSynchronize: 1,461 calls, 48,041 ms
       Avg 32.9 ms, max 3,079 ms
```

**48 seconds in hard CPU↔GPU syncs.** Root cause of 70.2% GPU density
(down from ~98.5% in run 9).

1,461 calls / 228 steps = ~6.4 DeviceSyncs per step. Likely sources:
- **cuDNN workspace virtual memory management** — `cuMemSetAccess` (217 calls) and
  `cuMemUnmap` (217 calls). cuDNN resizes workspace using VMM ops that require device idle.
- **Conv1d breaking CUDA graph capture** — forces sync at graph boundary every step.
- **DALI PopOutputs** — one sync per batch fetch.

## GPU Utilization

| Metric              | Value                                    |
|---------------------|------------------------------------------|
| Total kernel time   | 50.7 s                                   |
| GPU active span     | 72.2 s                                   |
| **Kernel density**  | **70.2%** (was ~98.5% in run 9)          |
| Total idle time     | 21.6 s across 188,291 gaps               |
| Gaps > 50µs         | 6,023 gaps — 20.1 s total               |
| Largest gaps        | 2.4 s, 842 ms, 807 ms, 578 ms…          |

Seconds-scale gaps = DeviceSync stalls. GPU sits idle while CPU waits.

## Kernel Breakdown

| Category              |  Time  | Share | Notes |
|-----------------------|-------:|------:|-------|
| Triton fused          | 18.0 s | 35.5% | LayerNorm, attention proj, MLP |
| GEMM                  | 13.9 s | 27.5% | Linear layers |
| FlashAttn bwd         |  7.7 s | 15.3% | |
| Other                 |  5.9 s | 11.6% | |
| Conv1d (token inter.) | ~5.2 s | ~10%  | nchwToNhwc + wgrad + depthwise fwd/bwd |
| FlashAttn fwd         |  2.9 s |  5.7% | |
| Elementwise           |  0.6 s |  1.2% | CUDA graphs capturing these ✓ |
| Optimizer             |  0.6 s |  1.1% | |
| Fill/zero_grad        |  0.5 s |  1.0% | |

Elementwise dropped from 23.7% (run 15) to 1.2% — CUDA graphs are working and
capturing those kernels successfully (~29 ms real-time saving confirmed).

## DALI Overhead

```
PopOutputs:            228 calls, 5.9 s,  avg 26.1 ms/call
ExternalSource:        229 calls, 4.8 s,  avg 20.8 ms/call
SetDataSource:         229 calls, 4.4 s,  avg 19.4 ms/call
cudaStreamSynchronize: 442,232 calls, 20.7 s — DALI internal multi-stream sync
```

~15 s of CPU-side DALI work per epoch. Partially overlaps GPU but the 442K
stream syncs are DALI-internal and not directly controllable.

## CUDA Graphs Status

- **454 graph launches** (2/step × 228 steps) — graphs active ✓
- **119,112 individual cudaLaunchKernel** — ops outside graph (DALI, masking, Conv1d)
- Elementwise at 1.2% confirms graph is capturing fill/copy/add ops ✓

## Verdict on TokenInteraction Conv1d

Conv1d adds ~10% GPU compute but causes ~28% GPU idle time through DeviceSync stalls.
Net effect on throughput is strongly negative.

## Options

1. **Replace Conv1d with Triton/pure-PyTorch kernel** that avoids cuDNN workspace
   management — avoids VMM syncs, stays inside CUDA graph.
2. **Remove token interaction** and check whether accuracy benefit justifies ~30%
   throughput loss.
3. **Try `torch.backends.cudnn.benchmark = True`** — may stabilize workspace selection
   and eliminate repeated resizing syncs (cuMemSetAccess/cuMemUnmap).
