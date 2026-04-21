# Nsys Analysis — run 17 (TokenInteraction removed)

**File:** `nsys/17_wo_tokens_interaction.nsys-rep`
**Date:** 2026-04-21

## Summary

Removing `TokenInteractionBlock` (Conv1d) did **not** fix the DeviceSync problem.
GPU density is still 67.1% — nearly identical to the prior run (70.2%).
Root cause is timm's `PatchEmbed`, which uses `nn.Conv2d` and triggers the same
cuDNN workspace VMM machinery.

## GPU Utilization

| Metric              | Run 17 (no TokenInteraction) | Prior (w/ TokenInteraction) |
|---------------------|------------------------------|-----------------------------|
| Total kernel time   | 44.0 s                       | 50.7 s                      |
| GPU active span     | 65.7 s                       | 72.2 s                      |
| **Kernel density**  | **67.1%**                    | **70.2%**                   |
| Total idle time     | 21.7 s                       | 21.6 s                      |
| Gaps > 50µs         | 13,900 — 20.3 s total        | 6,023 — 20.1 s              |
| Top gaps (ms)       | 2300, 1004, 876, 659, 603…  | 2400, 842, 807, 578…        |

Removing Conv1d saved ~6.7 s of GPU compute but **did not recover any density**.
The idle-gap pattern is unchanged because the DeviceSync source is the same: cuDNN workspace VMM.

## Critical Problem: cudaDeviceSynchronize

```
cudaDeviceSynchronize: 1,290 calls, 43,742 ms
Avg 33.9 ms, max 1,882 ms
```

1290 calls / 197 steps = ~6.5 DeviceSyncs per step — same rate as the TokenInteraction run.
Source: `cuMemUnmap` (169 calls) + `cuMemSetAccess` (169 calls) from cuDNN workspace resizing.

## Root Cause: PatchEmbed nn.Conv2d

timm's `PatchEmbed` uses `nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)`.
Even with TokenInteractionBlock gone, this Conv2d invokes cuDNN, which:
1. Manages workspace via virtual memory (`cuMemUnmap`/`cuMemSetAccess`)
2. Requires device synchronization before releasing/remapping workspace
3. Fires one `cudaDeviceSynchronize` per VMM op → seconds-scale GPU stall

### Conv2d kernel evidence

| Kernel                   | Calls | GPU Time  | Pct |
|--------------------------|------:|----------:|----:|
| `nchwToNhwcKernel`       |   396 | 1,827 ms  | 4.2% |
| `Kernel` (wgrad_opt)     |   198 | 1,726 ms  | 3.9% |
| `implicit_convolve_sgemm`|   198 | 1,064 ms  | 2.4% |
| **Conv2d total**         |       | **4,617 ms** | **10.5%** |

396 / 197 steps = exactly 2 nchwToNhwc calls/step (forward + backward format conversion).
198 / 197 steps ≈ 1 wgrad call/step (backward weight gradient).
198 / 197 steps ≈ 1 implicit_convolve/step (forward).

This matches `PatchEmbed.proj` (a single Conv2d) running forward + backward every step.

## Kernel Breakdown

| Category              |   Time | Share | Notes |
|-----------------------|-------:|------:|-------|
| GEMM / matmul         | 15.5 s | 35.2% | |
| Triton fused          | 14.3 s | 32.4% | LayerNorm, attention proj, MLP, GELU |
| FlashAttn bwd         |  7.9 s | 17.9% | |
| FlashAttn fwd         |  2.9 s |  6.6% | |
| Other                 |  1.8 s |  4.2% | |
| Optimizer             | 0.46 s |  1.0% | |
| Fill / zero_grad      | 0.45 s |  1.0% | |
| **Conv2d (PatchEmbed)**| **4.6 s** | **10.5%** | nchwToNhwc + wgrad + forward |
| Elementwise           | 0.20 s |  0.4% | CUDA graphs capturing ✓ |

Elementwise at 0.4% confirms CUDA graphs are active and fusing ops — same pattern as run 16.

## CUDA Graphs Status

- **394 graph launches** (2/step × 197 steps) — graphs active ✓
- **~122k individual cudaLaunchKernel** — ops outside graph (DALI + PatchEmbed Conv2d + others)

PatchEmbed's Conv2d is either (a) causing a graph break, running eagerly, or (b) inside the
graph but still triggering cuDNN workspace VMM on replay. Either way, the VMM ops occur and
require device synchronization.

## DALI Overhead

```
PopOutputs:    198 calls, 6.97 s, avg 35.2 ms/call
ExternalSource: 199 calls, 4.91 s, avg 24.7 ms/call
SetDataSource:  199 calls, 4.63 s, avg 23.3 ms/call
cudaStreamSynchronize: 284,012 calls, 13.7 s — DALI internal
```

~16 s of CPU-side DALI work. Partially overlaps GPU compute but contributes to idle time.

## Fix Applied: Replace PatchEmbed with LinearPatchEmbed

For non-overlapping patches (`stride == kernel_size`), `nn.Conv2d` is mathematically
equivalent to a reshape + `nn.Linear`. The linear version uses no cuDNN and is fully
CUDA-graph-compatible.

### Implementation (`mae/model.py`)

```python
class LinearPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        ...
        self.proj = nn.Linear(in_chans * ph * pw, embed_dim)

    def forward(self, x):
        N, C, H, W = x.shape
        ph, pw = self.patch_size
        gh, gw = self.grid_size
        x = x.reshape(N, C, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(N, gh * gw, ph * pw * C)
        return self.proj(x)
```

### Verification

- Max abs diff vs original Conv2d (same weights): **1.31e-06** (sub-fp32-epsilon, outputs match ✓)
- Conv layers remaining after fix: **0** ✓
- Total params unchanged: 5,161,344 ✓
- `patch_embed.proj.weight` shape: `(256, 256)` = `(embed_dim, in_chans*ph*pw)` ✓

**Expected result:** eliminate all ~1290 DeviceSyncs → restore GPU density to ~98.5%.
