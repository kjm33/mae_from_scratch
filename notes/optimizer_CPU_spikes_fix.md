# Optimizer CPU Spikes — Root Cause and Fix

## Problem

Earlier runs (bitsandbytes 8-bit optimizer) showed dozens of `cudaDeviceSynchronize`
calls per step during the optimizer phase — one per parameter update. Each sync stalls
the CPU until the GPU finishes that parameter's kernel, serializing all optimizer work
and causing high CPU usage spikes.

## Fix

Two changes eliminated the problem:

1. **Switched to standard `AdamW(fused=True)`** — single fused kernel over all parameters
   via `multi_tensor_apply_kernel`, no per-parameter syncs.

2. **Compiled the full `train_step` with `torch.compile`** (including optimizer) — Inductor
   captures forward, backward, and optimizer into CUDA graphs. The CPU only issues two
   `cudaGraphLaunch` calls per step and is otherwise free.

## Evidence (run 5 steady-state trace)

```
CUDA RUNTIME CALLS
  1.7 ms | x2 | avg 854 us | cudaGraphLaunch     ← entire step: fwd+bwd+optimizer
  94 us  | x16| avg 5.9 us | cudaLaunchKernel    ← minor non-graph kernels only
```

```
PHASE BREAKDOWN (GPU)
  70.5% | 704 kernels | 138.6 ms  → train_step graph   (forward + backward)
  29.5% | 303 kernels |  58.1 ms  → CompiledFxGraph     (optimizer)
```

CPU-side optimizer overhead per step:
- `zero_grad`: 115 µs
- Graph launch: ~854 µs
- **Total: ~1 ms** — vs dozens of blocking syncs before

GPU optimizer time: 58ms (`multi_tensor_apply_kernel` ×8, 9.5ms + surrounding ops).
GPU kernel density: 99.7%. No idle gaps attributable to optimizer.

## Remaining sync (now also fixed)

The only `cudaDeviceSynchronize` still visible in the trace (187ms) was from
`loss.item()` called every step. Fixed by accumulating loss on GPU and syncing
once per epoch instead.
