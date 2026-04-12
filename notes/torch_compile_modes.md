# torch.compile Modes: default vs reduce-overhead vs max-autotune

**Date:** 2026-04-12
**Context:** MAE ViT-Base training, RTX 3090, full train_step compiled (forward + backward + optimizer)

---

## Mode Comparison

| Mode | What it does | Compile time | Runtime |
|------|-------------|-------------|---------|
| `default` | Fuses ops via TorchInductor (Triton/C++ kernels) | Fast | Good |
| `reduce-overhead` | Same as default **+ CUDA graphs** | Medium | Better (less CPU dispatch) |
| `max-autotune` | Same as reduce-overhead **+ kernel autotuning** | Slow | Best |

## default

Uses TorchInductor to trace the computation graph and generate optimized Triton and C++ kernels. Fuses compatible operations (e.g., LayerNorm + residual add, GELU + its backward) into single kernels to reduce memory traffic and kernel launch count. Does not use CUDA graphs — each step still dispatches kernels individually from the CPU.

## reduce-overhead

Everything `default` does, plus wraps the execution in CUDA graphs. Instead of the CPU launching hundreds of kernels per step, it records the entire kernel sequence once and replays it with a single `cudaGraphLaunch` call. This is what eliminated the CPU dispatch overhead in this project (optimizer went from 252 individual launches to a single graph replay).

## max-autotune

Everything `reduce-overhead` does, plus benchmarks multiple kernel implementations for each operation and picks the fastest. For each GEMM, it trials different tile sizes, data layouts (TN, NN, NT), and backends (cuBLAS, cuDNN, Triton). Since 68% of GPU time in our workload is GEMMs, finding optimal tilings for the specific matrix shapes (determined by batch_size=256, embed_dim=768, patch count, etc.) can yield measurable speedups.

The autotuning results are cached in `.cache/` (set via `TORCHINDUCTOR_CACHE_DIR`), so the cost is paid only once. Subsequent runs reuse the cached kernel selections.

## Why max-autotune is the right choice here

1. **GEMM-dominated workload (68% of GPU time)** — autotuning has the most impact when matmuls dominate, because there are many valid GEMM kernel configurations and the optimal one depends on the exact matrix dimensions.
2. **Fixed shapes throughout training** — batch_size=256, img_size=(32, 512), mask_ratio=0.75 never change. CUDA graphs and autotuned kernels both require static shapes.
3. **Long training runs** — multi-epoch pretraining amortizes the one-time compile cost (minutes) into nothing.
4. **Results are cached** — the `.cache/` directory stores autotuning results, so only the very first run pays the compile cost.

## CUDA Graph Constraints

Both `reduce-overhead` and `max-autotune` use CUDA graphs, which require:

- **Static tensor shapes** — every tensor must have the same shape on every step. No dynamic batch sizes (use `drop_last=True` on the DataLoader if the last batch might be smaller).
- **No CPU-side branching on tensor values** — control flow can't depend on tensor content (e.g., no `if loss > threshold`).
- **Static random state** — random operations (like `torch.randperm` for masking) work fine; the RNG state advances on each graph replay. The randomness is different each step.
- **Fixed memory addresses** — tensors are replayed at the same GPU addresses. Allocations inside the graph must be consistent.

## Compile Time Expectations

| Mode | First step | Cached runs |
|------|-----------|-------------|
| `default` | ~10-30 seconds | ~5-10 seconds |
| `reduce-overhead` | ~30-60 seconds | ~10-15 seconds |
| `max-autotune` | ~2-5 minutes | ~10-15 seconds |

These are rough estimates. Actual times depend on model complexity and number of unique operations to tune.
