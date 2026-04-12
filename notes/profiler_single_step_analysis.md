# Profiler Analysis: Single Training Step

**Trace file:** `runs/4_single_step/T-1000_13303.1775939688285173222.pt.trace.json`
**Date:** 2026-04-12
**Setup:** ViT-Base MAE (patch8, 32x512), batch_size=256, bf16 autocast, torch.compile(mode="reduce-overhead"), bitsandbytes AdamW8bit, RTX 3090

---

## 1. Step Timing Breakdown

| Phase | GPU Time | CPU Time | Dispatch Method | Notes |
|-------|----------|----------|-----------------|-------|
| **Forward** | 61.7 ms | 2.8 ms | CUDA graph (2 graph launches) | Fully compiled, near-zero CPU overhead |
| **Backward** | 136.1 ms | 2.6 ms | CUDA graph | Fully compiled, near-zero CPU overhead |
| **Optimizer (AdamW8bit)** | 24.2 ms | 24.8 ms | 252 individual kernel launches | 252 `cudaDeviceSynchronize` calls |
| **Total** | **~222 ms** | **227.5 ms** | | |

- Total events in trace: 45,558
- Total GPU kernel count: 1,513
- Forward kernels: 305, Backward kernels: 957, Optimizer kernels: 251

---

## 2. GPU Utilization

| Metric | Value |
|--------|-------|
| GPU active span | 225.8 ms |
| Total GPU kernel time | 204.7 ms |
| **Kernel density** | **90.7%** |
| Total GPU idle gaps | 21.1 ms |
| Number of gaps | 743 |
| Average gap | 28.5 us |
| Largest gap | 286 us |
| GPU memcpy time | 8 us (negligible) |
| GPU memset time | 72 us (negligible) |

**Interpretation:** 90.7% kernel density is very good. The GPU is almost never idle during the step. The 21 ms of idle gaps are spread across 743 tiny gaps (avg 28.5 us), which is normal inter-kernel scheduling overhead. There are no large stalls or data transfer bottlenecks.

---

## 3. GPU Kernel Breakdown by Type

### Top 20 Kernels (aggregated by name)

| Kernel | Total Time | % of GPU | Count | Category |
|--------|-----------|----------|-------|----------|
| GEMM bf16 128x256 (cutlass, relu) | 31,288 us | 15.3% | 60 | matmul (backward) |
| GEMM bf16 256x128 (cutlass, relu) | 20,441 us | 10.0% | 26 | matmul (backward) |
| GEMM bf16 256x128 (ampere, relu, tn) | 19,125 us | 9.3% | 28 | matmul (backward) |
| FlashAttention backward (dq_dk_dv) | 16,104 us | 7.9% | 20 | attention backward |
| GEMM bf16 128x128 (ampere, nt) | 10,853 us | 5.3% | 24 | matmul |
| GEMM bf16 128x128 (ampere, nn) | 9,701 us | 4.7% | 34 | matmul |
| GEMM bf16 128x128 (ampere, tn) | 8,627 us | 4.2% | 16 | matmul |
| GEMM bf16 256x128 (ampere, nn) | 7,557 us | 3.7% | 8 | matmul (forward) |
| GEMM bf16 128x256 (ampere, nt) | 6,895 us | 3.4% | 8 | matmul (forward) |
| GEMM bf16 128x256 (ampere, nt) | 6,872 us | 3.4% | 8 | matmul |
| Triton fused LayerNorm+backward | 6,199 us | 3.0% | 39 | normalization |
| GEMM bf16 128x128 (ampere, relu, tn) | 5,725 us | 2.8% | 12 | matmul |
| 8-bit optimizer blockwise update | 5,072 us | 2.5% | 83 | optimizer |
| FlashAttention forward | 4,781 us | 2.3% | 8 | attention forward |
| Triton fused LayerNorm backward | 4,266 us | 2.1% | 22 | normalization |
| Triton fused LayerNorm backward | 3,693 us | 1.8% | 15 | normalization |
| Triton fused GELU backward | 3,237 us | 1.6% | 8 | activation |
| Triton fused GELU forward | 2,194 us | 1.1% | 8 | activation |
| GEMM bf16 256x128 (ampere, nt) | 2,022 us | 1.0% | 9 | matmul |
| FlashAttention backward (dot_do_o) | 1,901 us | 0.9% | 8 | attention backward |

### Summary by Category

| Category | Total Time | % of GPU |
|----------|-----------|----------|
| **GEMM (matmul) bf16** | ~108 ms | **52.7%** |
| **FlashAttention backward** | ~18 ms | **8.8%** |
| **Triton fused (LayerNorm + backward)** | ~14.1 ms | **6.9%** |
| **FlashAttention forward** | ~4.8 ms | **2.3%** |
| **Triton fused (GELU + backward)** | ~5.4 ms | **2.7%** |
| **8-bit optimizer blockwise** | ~5.1 ms | **2.5%** |

The model is **GEMM-dominated** (52.7% in matrix multiplications), which means tensor cores are being utilized effectively. bf16 tensor core GEMMs are the most efficient operation for this hardware.

---

## 4. CPU Analysis

### Top CPU Operations (aggregated)

| Operation | Total Time | % of CPU | Count | Notes |
|-----------|-----------|----------|-------|-------|
| AccumulateGrad (evaluate_function) | 192,822 us | 32.1% | 252 | **Misleading — see below** |
| AccumulateGrad | 192,259 us | 32.0% | 252 | Same as above |
| aten::add_ | 191,894 us | 31.9% | 252 | Gradient accumulation |
| Torch-Compiled Region | 5,124 us | 0.9% | 1 | Graph dispatch |
| bitsandbytes::optimizer_update_32bit | 4,904 us | 0.8% | 169 | Optimizer |
| CompiledFunction | 4,740 us | 0.8% | 1 | Graph dispatch |
| CompiledFunctionBackward | 3,027 us | 0.5% | 1 | Graph dispatch |
| bitsandbytes::optimizer_update_8bit | 2,659 us | 0.4% | 83 | Optimizer |

### The 190 ms `AccumulateGrad` Explained

The single largest `AccumulateGrad` call (190,344 us) is **not a real bottleneck**. It contains a single `cudaLaunchKernel` call that blocks for 190 ms because the CUDA stream is busy executing the backward CUDA graph. The CPU dispatched the backward graph in 2.6 ms, then has nothing to do while the GPU crunches through 957 backward kernels over 136 ms. The CPU is blocked inside `cudaLaunchKernel` waiting for stream capacity. The remaining 251 `AccumulateGrad` calls are tiny (avg < 1 us each).

### CUDA Runtime Overhead

| Runtime Call | Total Time | Count | Avg per Call |
|-------------|-----------|-------|-------------|
| cudaLaunchKernel | 192,767 us | 509 | 378.7 us |
| cudaDeviceSynchronize | 3,960 us | 253 | 15.7 us |
| cudaGraphLaunch | 1,201 us | 2 | 600.5 us |
| cudaMemcpyAsync | 24 us | 2 | 11.9 us |

The 509 `cudaLaunchKernel` calls cost 378.7 us average, but most of that average is inflated by the single 190 ms blocking call. The 2 `cudaGraphLaunch` calls (forward + backward CUDA graphs) are highly efficient at 600 us each to dispatch hundreds of kernels.

---

## 5. Identified Bottlenecks

### Bottleneck #1: bitsandbytes AdamW8bit — Per-Parameter Synchronization

**Severity: Medium (~11% of step time is overhead)**

The optimizer is the only phase not captured by CUDA graphs. bitsandbytes calls `cudaDeviceSynchronize` after every single parameter update:

- 252 `cudaLaunchKernel` calls (one per parameter)
- 252 `cudaDeviceSynchronize` calls (one per parameter)
- 6 ms of actual GPU compute stretched to 24.8 ms wall time
- **~18.8 ms of pure synchronization overhead**

This forces full CPU-GPU serialization 252 times per step. Each sync is cheap (15.7 us avg), but they prevent any kernel pipelining in the optimizer phase.

### Bottleneck #2: Backward Pass Dominates GPU Time

**Severity: Informational (structural, not fixable)**

The backward pass (136.1 ms) is 2.2x the forward pass (61.7 ms). This is expected in MAE because:
- The encoder processes only 25% of patches (forward is cheap)
- The decoder reconstructs all patches
- Backward needs gradients for both encoder and decoder, plus the attention backward kernels are more expensive than forward (flash_bwd_dq_dk_dv at 1,580 us each vs flash_fwd at 598 us each — 2.6x ratio)

### Bottleneck #3: No CPU-GPU Overlap During Backward

**Severity: Low**

While the backward CUDA graph runs on GPU for 136 ms, the CPU is blocked inside `cudaLaunchKernel` for 190 ms (waiting for stream capacity). There is no useful CPU work happening in parallel. In a multi-step training loop, this would overlap with data loading from the next batch if using a prefetch pipeline — but the in-RAM dataset already eliminates data loading cost, so there is nothing for the CPU to do.

---

## 6. What Is Already Working Well

- **torch.compile with CUDA graphs** — Forward and backward are fully graph-captured. The CPU dispatches each phase in ~2.7 ms regardless of how many kernels run on GPU. This is the gold standard for dispatch efficiency.
- **FlashAttention (SDPA)** — Both forward and backward use flash attention kernels. Forward attention is only 2.3% of GPU time.
- **bf16 autocast** — All GEMMs use bf16 tensor cores (ampere_bf16_s16816gemm variants). This is optimal for RTX 3090 (Ampere).
- **In-RAM dataset + pin_memory** — Zero data loading overhead visible in the trace. Only 8 us of GPU memcpy total.
- **Triton fused kernels** — LayerNorm, GELU, and their backward passes are fused by torch.compile into efficient Triton kernels.
- **90.7% GPU kernel density** — The GPU is almost never idle.

---

## 7. Optimization Recommendations

### Recommendation 1: Replace AdamW8bit with fused AdamW (Priority: High)

Switch from bitsandbytes to PyTorch's built-in fused optimizer:

```python
# Before (252 syncs per step, 24.8 ms):
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1.5e-4, weight_decay=0.05)

# After (single fused kernel, no syncs):
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, fused=True)
```

**Why:** `fused=True` runs the entire optimizer step in a single fused CUDA kernel per parameter group, eliminating all 252 synchronization points. Available since PyTorch 2.0 for CUDA tensors.

**Trade-off:** Uses fp32 optimizer states (~600 MB more VRAM than 8-bit). On an RTX 3090 with 24 GB, this should be acceptable. Expected speedup: ~10% per step (24.8 ms -> ~6 ms optimizer phase).

### Recommendation 2: Increase Batch Size (Priority: High)

```python
# Try progressively:
batch_size = 384  # or 512 if VRAM allows
```

**Why:** Each training step has fixed overhead (CUDA graph launch, optimizer step, inter-kernel gaps). Larger batches amortize this fixed cost over more samples. The GEMM kernels are already using large tile sizes (256x128), but more data per kernel improves arithmetic intensity. Additionally, FlashAttention forward is only 4.8 ms (sequence length of 256 visible tokens is short), meaning attention cost won't scale badly.

**Trade-off:** Higher VRAM usage for activations. Profile after switching to `fused=True` AdamW to see available headroom.

### Recommendation 3: Compile the Entire Training Step (Priority: Medium)

```python
@torch.compile(mode="reduce-overhead")
def train_step(model, batch, optimizer):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _, _ = model(batch, mask_ratio=0.75)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss
```

**Why:** Currently, only the model forward+backward are compiled into CUDA graphs. The optimizer runs eagerly with 252 individual kernel launches. Wrapping the entire step lets torch.compile capture the optimizer too, potentially folding it into the graph.

**Caveat:** This requires the standard PyTorch AdamW (not bitsandbytes). The bitsandbytes optimizer is opaque to torch.compile.

### Recommendation 4: Gradient Accumulation (Priority: Low)

```python
accum_steps = 2  # effective batch = 256 * 2 = 512

for step, batch in enumerate(dataloader):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _, _ = model(batch, mask_ratio=0.75)
        loss = loss / accum_steps
    loss.backward()
    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

**Why:** If VRAM doesn't allow larger batches, gradient accumulation achieves the same effective batch size without extra activation memory. This cuts optimizer overhead in half (runs every 2 steps). The forward+backward still run every step but the expensive optimizer sync overhead is halved.

---

## 8. Expected Impact Summary

| Change | Est. Step Time | Speedup | VRAM Impact |
|--------|---------------|---------|-------------|
| Current (baseline) | 227.5 ms | — | — |
| Fused AdamW (drop bnb) | ~209 ms | ~8% | +600 MB |
| Fused AdamW + batch 384 | ~235 ms/step but 50% more samples | ~50% throughput | +2-3 GB |
| Fused AdamW + batch 512 | ~260 ms/step but 2x samples | ~75% throughput | +4-5 GB |
| + Compile full step | additional ~5 ms savings | marginal | none |

> **Key insight:** The single biggest throughput gain comes from increasing batch size, not reducing per-step overhead. The pipeline is already well-optimized — 90.7% GPU utilization with CUDA graphs and FlashAttention. The main lever left is putting more work per step on the GPU.
