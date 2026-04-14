# Profiler Analysis — Run 9: Ultra-Slim Model (patch 32×8)

Trace: `runs/9_ultra_light_model_2026_04_14__19_09_42.pt.trace.json`
Model: `mae_vit_ultra_slim` — embed_dim=256, depth=6, num_heads=4, decoder embed=128, depth=2; patch_size=(32,8) → 64 patches; 108 learnable parameters.

**Overall: excellent health — 10ms total GPU step time, 98.5% kernel density.**

---

## GPU Utilization

- Kernel density: **98.5%** (466 kernels, only 175µs idle total, one gap of 146µs)
- 2× `cudaGraphLaunch` — CUDA graphs active for forward+backward and optimizer

## Phase Breakdown

| Phase | GPU time | Share |
|---|---|---|
| forward+backward graph (`fxyw...`) | 7.3ms | 73% |
| optimizer compiled graph (`f5wq...`) | 2.7ms | 27% |

## Kernel Type Breakdown

| Type | GPU time | Share |
|---|---|---|
| Triton fused (torch.compile) | 4.4ms | 44.5% |
| GEMM / matmul | 2.4ms | 24.5% |
| FlashAttention backward | 1.5ms | 15.4% |
| Optimizer (`multi_tensor_apply`, 3 calls) | 1.0ms | 10.3% |
| CuDNN patch embed conv (32×8 kernel) | ~500µs | 3.6% |

## Notable Points

- **Single `cudaDeviceSynchronize` at 5.4ms** — from profiler `export_chrome_trace` flush, not real training overhead.
- Two CuDNN conv kernels for patch embedding (new rectangular (32,8) kernel shape): `cutlass_cudnn` + `implicit_convolve_sgemm`.
- 108 `AccumulateGrad` calls on CPU during backward — expected, fine at this scale.
- No memcpy, no memset, no excessive individual kernel launches.

## Bottom Line

Compute-bound in a healthy way. FlashAttention backward is the largest single consumer (15%). With only 64 patches (vs 256 with square patch_size=8), attention compute is minimal — Triton fusions dominate, which is ideal.
