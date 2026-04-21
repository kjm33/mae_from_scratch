# NCU Analysis — run 16 (max-autotune + CUDA graphs)

**File:** `ncu/16_with_autotune_CUDA_graphs.ncu-rep`
**Date:** 2026-04-21

117.4 ms total GPU time, 4,957 kernel invocations.
*(NCU serializes kernels — real step time is ~10 ms)*

## Time Breakdown by Category

| Category        |  Time  | Share | Avg SM% |
|-----------------|-------:|------:|--------:|
| GEMM            | 29.7 ms | 25.3% | 30.2% |
| Elementwise     | 29.2 ms | 24.8% | 11.3% |
| LayerNorm       | 14.6 ms | 12.4% | 40.8% |
| FusedAdamW      | 10.9 ms |  9.3% | 23.3% |
| FlashAttn bwd   | 10.5 ms |  8.9% | 21.0% |
| Reduce          |  5.9 ms |  5.0% |  6.9% |
| Other           |  4.6 ms |  3.9% | 52.2% |
| FlashAttn fwd   |  4.4 ms |  3.8% | 35.0% |
| GELU            |  2.7 ms |  2.3% | 44.1% |
| Transpose       |  2.3 ms |  2.0% | 59.1% |
| Scatter/Gather  |  0.9 ms |  0.8% | 20.4% |
| GEMM split-K    |  0.7 ms |  0.6% | 15.4% |

## Comparison vs Run 15 (compile(default), no CUDA graphs)

| Category       | Run 15  | Run 16  |  Delta  | Delta%  |
|----------------|--------:|--------:|--------:|--------:|
| GEMM           | 31.9 ms | 29.7 ms | -2.2 ms |  -6.8% |
| Elementwise    | 29.7 ms | 29.2 ms | -0.5 ms |  -1.8% |
| LayerNorm      | 14.2 ms | 14.6 ms | +0.4 ms |  +2.5% |
| FlashAttn bwd  | 13.1 ms | 10.5 ms | -2.6 ms | -20.3% |
| FusedAdamW     | 11.6 ms | 10.9 ms | -0.7 ms |  -5.7% |
| FlashAttn fwd  |  5.1 ms |  4.4 ms | -0.7 ms | -12.9% |
| GELU           |  3.3 ms |  2.7 ms | -0.6 ms | -18.6% |
| Scatter/Gather |  1.2 ms |  0.9 ms | -0.4 ms | -28.9% |
| **TOTAL**      | **125.3 ms** | **117.4 ms** | **-7.9 ms** | **-6.3%** |

## Key Finding: CUDA Graphs Are Invisible to NCU

**Elementwise: 2,964 → 2,928 invocations, -1.8%** — essentially unchanged.

NCU cannot measure CUDA graph benefit. NCU replays each kernel in isolation,
eliminating CPU gaps between kernels. CUDA graphs only reduce CPU-side launch
overhead (Python/runtime dispatch time *between* kernels), which doesn't appear
in kernel execution time.

**The entire 7.9 ms improvement comes from `max-autotune` kernel selection**, not
from CUDA graphs:
- FlashAttn bwd: -20.3% — better backward kernel config selected
- FlashAttn fwd: -12.9% — better forward kernel config
- GELU: -18.6% — better elementwise fusion
- Scatter/Gather: -28.9% — better memory access pattern
- GEMM: -6.8% — better tile configuration

## To Measure CUDA Graph Benefit

NCU is the wrong tool. Use nsys or wall-clock timing to compare step time ms:
- `compile(reduce-overhead)` @ batch=5120 (CUDA graphs on)
- `compile(default)` @ batch=8192 (CUDA graphs off)

CUDA graphs eliminate CPU overhead for ~4,900 kernel launches/step. At ~5µs each,
that's ~25 ms of CPU launch latency eliminated — visible only in wall-clock time.

## Occupancy

- Weighted-avg SM throughput: 28.6% (identical to run 15)
- 50% of kernel invocations still at SM < 10% — model size is the ceiling
