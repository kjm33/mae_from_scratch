# nsys Analysis — Run 10 (`mae_vit_ultra_slim`, patch 32×8)

Trace: `nsys/10_slim_patch32x8.nsys-rep`

This captures the **full training run** (all epochs), unlike the single-step PyTorch trace.

---

## GPU Kernel Summary

**Top time consumers:**

| % | Total | Instances | Kernel |
|---|---|---|---|
| **21%** | 636ms | 78,158 | `vectorized_elementwise_kernel<FillFunctor<int>>` |
| 6% | 179ms | 504 | `multi_tensor_apply_kernel` (fused AdamW) |
| 4% | 119ms | 1,008 | FlashAttention backward (head=64) |
| 3% | 97ms | 336 | FlashAttention backward (head=32) |
| 3% | 91ms | 2,178 | Triton: layer norm + conv + gather (fused) |
| 2% | 83ms | 2,016 | GEMM `bf16 64×64` |

The **78,158 FillFunctor calls at 21%** is the biggest anomaly. At 504 optimizer steps that's ~155 fills per step — far too many to come from `zero_grad(set_to_none=True)`. This is coming from the random masking path (`torch.ones`, scatter/gather) and/or DALI internal fills, both fired outside the CUDA graph.

The **504 AdamW calls** = total optimizer steps. With 334 `cudaGraphLaunch` calls, the first ~170 steps ran without CUDA graphs (compilation warmup), the rest used graphs.

---

## CUDA API Summary — critical findings

| % | Total | Calls | Avg | API |
|---|---|---|---|---|
| **39%** | 1,813ms | 1,698 | 1.07ms | `cudaDeviceSynchronize` |
| **23%** | 1,084ms | 37,090 | 29µs | `cudaStreamSynchronize` |
| 13% | 589ms | 107,048 | 5.5µs | `cudaLaunchKernel` |
| 1% | 72ms | 334 | 216µs | `cudaGraphLaunch` |

**`cudaDeviceSynchronize` at 39%** is the headline issue — 1,698 calls averaging 1.07ms each, with a max of 161ms. That's ~3.4 hard CPU↔GPU syncs per optimizer step. At ~10ms GPU per step, each sync costs ~10% of step time. These are coming from DALI's internal pipeline stages, not training code.

**37,090 `cudaStreamSynchronize` calls** (~73/step) — entirely DALI: it runs ExternalSource, CropMirrorNormalize, and MakeContiguous on separate CUDA streams and syncs between them.

**107,048 `cudaLaunchKernel`** (~212/step) — most of these are outside the CUDA graphs (DALI ops, masking, zero_grad). The compiled forward+backward+optimizer runs as only **334 graph launches** total.

---

## NVTX / DALI Pipeline

| % | Instances | Avg | Max | Range |
|---|---|---|---|---|
| 27% | 170 | 702µs | **100ms** | DALI CropMirrorNormalize |
| 23% | 168 | 606µs | **100ms** | DALI PopOutputs |
| 15% | 170 | 396µs | 2.9ms | DALI ExternalSource |
| 9% | 170 | 250µs | 1.7ms | DALI SetDataSource |
| 2% | 170 | 75µs | — | H2D non-coalesced (MakeContiguous) |

The **100ms outlier stalls** on CropMirrorNormalize and PopOutputs are the biggest training-level issue. Median is 77µs (fine), but occasionally the OS evicts the memmap pages and the pipeline stalls while reloading them. These stalls pause the GPU entirely since the training loop blocks on `batch_data[0]["images"]` before launching the next CUDA graph.

The **H2D non-coalesced** label on MakeContiguous means DALI is copying data in a non-optimal memory layout — expected given the ExternalSource→GPU path with a numpy memmap source.

---

## Bottom Line

| Area | Status |
|---|---|
| GPU training kernels | Healthy — Triton-fused, CUDA graphs, FlashAttention |
| FillFunctor (21%) | Investigate — likely masking ops outside CUDA graph |
| DALI syncs (39%+23% of CUDA API) | Expected overhead from multi-stream pipeline |
| DALI 100ms stalls | Real issue — memmap page eviction, stalls GPU |
| CUDA graphs | Working — 334 graph launches after ~170 warmup steps |

The main opportunity is the DALI stalling: those 100ms outliers mean the GPU sits completely idle waiting for data. Pinning the memmap into RAM (`mlockall` or `numpy.load` with prefetch) or switching to a persistent in-memory tensor would eliminate them.
