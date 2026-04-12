# Profiler Analysis — Run 5 (optimizer_fused_and_compiled)

## Context

First epoch takes ~48s vs ~17s in previous runs. Two traces captured:

- `...620` — early training steps (profiler schedule active=3, steps 2–4 of epoch 1)
- `...350` — single `profile_step()` call after training (steady-state compiled step)

---

## Root Causes

### 1. `max-autotune` benchmarking — expected, ~2× first-epoch slowdown

During epoch 1, `torch.compile(mode="max-autotune")` benchmarks multiple GEMM tilings
and Triton kernel configs for every op. Early steps take **~417ms each**. After
compilation stabilises the same step takes **197ms** (steady state from `...350` trace).
That ~2× overhead applied to every step in epoch 1 accounts for most of the 48s vs 17s
difference. Results are cached in `.cache/` — subsequent runs pay no penalty.

### 2. `loss.item()` forces full CPU–GPU sync every step — fixable

```
# Trace ...620 — TOP CPU OPERATIONS
  47.1% |  1.21 s  | x3  | aten::item
  47.1% |  1.21 s  | x3  | aten::_local_scalar_dense

# CUDA RUNTIME
  1.21 s | x3 | avg 403ms | cudaStreamSynchronize
```

`logger.on_step(loss.item())` calls `cudaStreamSynchronize` — the CPU blocks until the
GPU finishes the full step before it can read the scalar. **403ms wasted per step** during
epoch 1 (autotuning steps are slower), **187ms in steady state** (from `cudaDeviceSynchronize`
in `...350` trace).

CUDA graphs reduce CPU dispatch to ~1ms — but `loss.item()` immediately negates that by
blocking until the GPU is done.

---

## Steady-State Step (after training, trace `...350`)

| Metric | Value |
|--------|-------|
| GPU kernel time | 196.7 ms |
| GPU kernel density | 99.7% |
| Idle gaps | 578 us total (231 gaps, avg 2.5 us) |
| CUDA graph launches | 2 |
| Kernel launches (non-graph) | 16 |

Two compiled graphs:
- `train_step` graph — 138.6ms, 704 kernels (70.5%) — forward + backward
- `CompiledFxGraph` — 58.1ms, 303 kernels (29.5%) — optimizer

GPU kernel breakdown (steady state):
- 50.7% Triton fused kernels
- 33.8% GEMM (matmul)
- 9.9%  FlashAttention backward
- 4.8%  Optimizer

GPU utilization is essentially perfect in steady state. The only remaining overhead is
the `loss.item()` sync on the CPU side.

---

## Fix: Stop syncing every step

Do not call `loss.item()` (or display it) every step. Options:

1. **Sync once per epoch** — accumulate loss as a GPU tensor, call `.item()` in `end_epoch()`
2. **Sync every N steps** — e.g. every 10 steps
3. **Async D2H copy** — `_loss_buf.copy_(loss.detach(), non_blocking=True)` so the CPU
   reads last step's loss while the GPU is already running the current step

Any of these would eliminate the `cudaStreamSynchronize` bottleneck and allow the CPU
to pipeline graph launches ahead of the GPU.
