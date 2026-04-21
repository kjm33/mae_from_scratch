# GPU Training Optimization Report
**Source:** *AI Systems Performance Engineering* — Chris Fregly, O'Reilly 2025  
**Scope:** Best practices applied to `mae_vit_ultra_light` on RTX 3090, as of 2026-04-21

---

## Status: What Is Already Done

| Practice | Status | Where |
|---|---|---|
| BF16 autocast | ✅ done | `train.py` — `autocast(dtype=bfloat16)` |
| FlashAttention via SDPA | ✅ done | `mae/model.py` — `FlashAttention` module |
| FusedAdamW | ✅ done | `train.py` — `AdamW(fused=True)` |
| DALI memmap pipeline (GPU-side decode) | ✅ done | `mae/dali_loader.py` |
| Loss sync once per epoch, not per step | ✅ done | `training_logger.py` — `.item()` in `end_epoch` |
| `torch.compile` | ✅ done | mode=`default` currently |
| F.layer_norm fused kernel in `norm_pix_loss` | ✅ done | `mae/model.py:301` |
| `.expand` instead of `.repeat` in gather ops | ✅ done | `mae/model.py:245,277` |
| Removed TokenInteractionBlock Conv1d | ✅ done | caused 48s DeviceSync stalls |

---

## Priority 1 — Switch to `max-autotune`

**Book reference:** Ch13 pp.548-553  
**Evidence:** NCU run 16 showed -7.9 ms total GPU time (-6.3%) vs run 15 with `default` mode.

`max-autotune` benchmarks multiple Triton tile configs and GEMM strategies on the first run,
caches the winners to `.cache/`, and reuses them on every subsequent run. All gains are
one-time cost. Measured improvements at this model's shapes:

| Kernel | Run 15 (default) | Run 16 (max-autotune) | Delta |
|---|---|---|---|
| FlashAttn bwd | 13.1 ms | 10.5 ms | -20.3% |
| FlashAttn fwd | 5.1 ms | 4.4 ms | -12.9% |
| GELU | 3.3 ms | 2.7 ms | -18.6% |
| Scatter/Gather | 1.2 ms | 0.9 ms | -28.9% |
| GEMM | 31.9 ms | 29.7 ms | -6.8% |

**Action:** In `train.py`, change one line:
```python
# Before:
@torch.compile(mode="default")
# After:
@torch.compile(mode="max-autotune")
```
Delete `.cache/` to force fresh autotuning if shapes have changed.

---

## Priority 2 — Benchmark CUDA Graphs vs Larger Batch

**Book reference:** Ch13 pp.554-568  
**Background:** In commit `073cb68`, CUDA graphs were traded off for a larger batch (5120→8192).
The book confirms CUDA graphs eliminate CPU launch overhead for every kernel (~5µs each × ~5000
kernels/step = ~25 ms of CPU overhead eliminated). This is **invisible to NCU** — only visible
in wall-clock time via `nsys` or `time.perf_counter`.

**The open question (from prior NCU analysis):** Does 60% larger batch (8192 vs 5120) compensate
for losing ~25 ms of CPU overhead amortization per step?

**Recommended experiment (takes ~10 minutes):**
```python
# Option A (current): compile(default), batch=8192
# Option B:          compile(reduce-overhead), batch=5120

# Measure wall-clock for 200 steps each, compare samples/sec
import time
t0 = time.perf_counter()
for _ in range(200):
    train_step(batch)
torch.cuda.synchronize()
samples_per_sec = 200 * batch_size / (time.perf_counter() - t0)
```

**Book guidance:**
- Capture as much of the training loop as possible into the graph (forward + backward + optimizer)
- Allocate all tensors before capture; no dynamic allocation inside the graph
- With static batch sizes, CUDA graphs are ideal — this model has fully static shapes

**Verdict:** If Option B (graphs on) wins in samples/sec, switch. The current setup was chosen
without a proper wall-clock comparison.

---

## Priority 3 — Remove `clip_grad_norm_`

**Book reference:** Ch13 p.599 — "Avoid synchronization gotchas"  
**Evidence (NCU run 15):** `GradNorm` 0.27 ms + `MultiTensor` 0.45 ms + several small-SM reduces
per step. Each `clip_grad_norm_` call forces a blocking CPU↔GPU sync (`.item()` on the norm).

**Action:** Remove from `train.py:124`:
```python
# Remove this:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

`OneCycleLR` with cosine decay and a 5M-param model has no practical gradient explosion risk.
The MAE paper (He et al. 2022) does not use grad clipping.

**Benefit:** Eliminates one `cudaDeviceSynchronize` per step + ~0.7 ms of kernel time.

---

## Priority 4 — CUDA Memory Allocator Tuning

**Book reference:** Ch13 pp.570-571  
**Relevant if:** VRAM usage grows over epochs or if OOM errors appear with larger batch sizes.

The PyTorch caching allocator can fragment memory when allocation sizes vary across steps.
The book recommends tuning with `PYTORCH_ALLOC_CONF`:

```bash
export PYTORCH_ALLOC_CONF=\
max_split_size_mb:256,\
roundup_power2_divisions:[256:1,512:2,1024:4,>:8],\
backend:cudaMallocAsync
```

- `max_split_size_mb:256` — keeps large free blocks intact; reduces fragmentation
- `roundup_power2_divisions` — standardizes allocation sizes into buckets; improves cache reuse
- `backend:cudaMallocAsync` — NVIDIA async allocator; avoids sync on memory events

**Note for this project:** `training_logger.py` reports VRAM via `mem_get_info` (driver-level
free vs. total). The "growing VRAM" seen across epochs is allocator fragmentation: total
allocated memory stays flat, but free contiguous blocks shrink. `cudaMallocAsync` can help
stabilize this.

**Monitor fragmentation:**
```python
stats = torch.cuda.memory_stats()
# If reserved_bytes stays high but allocated_bytes drops, fragmentation is growing
print(stats["reserved_bytes.all.current"] - stats["allocated_bytes.all.current"])
```

---

## Priority 5 — Avoid Synchronization Gotchas

**Book reference:** Ch13 pp.599, 627  
**Principle:** Any CPU access to a GPU tensor value causes an implicit `cudaStreamSynchronize`.

Common accidental syncs:
```python
# BAD — .item() syncs GPU every call:
loss_val = loss.item()  # inside training loop

# GOOD — sync once per epoch (already done in this project):
epoch_loss += loss.detach()        # stays on GPU
avg_loss = epoch_loss.item() / N   # one sync at epoch end
```

Other sync traps to avoid:
```python
# BAD:
if loss > threshold:        # Python bool from GPU tensor → sync
    ...
norm = grads.norm().item()  # explicit .item() → sync

# GOOD:
# Use torch.cuda.Event for GPU timing instead of time.time()
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record(); ...; end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

**Status:** The project already avoids per-step `.item()` in loss accumulation. No further
action needed unless new code is added.

---

## Priority 6 — Verify Tensor Cores Are Being Used

**Book reference:** Ch13 pp.599-600  
**Principle:** `autocast` selects compute dtypes but does not cast weight storage. If a layer
receives FP32 input that bypasses autocast, it falls back to full FP32 GEMM.

**Already verified:** NCU run 15 confirmed bf16 tensor-core GEMMs (222 regs/thread is the
Ampere tensor-core bf16 signature). Tensor Cores are active.

**For TF32 workloads (not applicable here — we use bf16):**
```python
torch.set_float32_matmul_precision("high")  # enables TF32 on Tensor Cores for FP32 ops
```

---

## Graph Breaks — What to Avoid

**Book reference:** Ch14 pp.608-612  
**Principle:** A graph break terminates TorchDynamo's current compiled subgraph and forces
fallback to eager execution, losing all kernel-fusion benefits for that region.

**Common graph-break causes:**
- `if tensor.item() > 0:` — Python bool from GPU tensor
- `print(tensor)` — CPU-side data access
- `len(tensor)` — triggers `.item()` internally
- cuDNN convolutions with variable workspace (the TokenInteractionBlock issue, now fixed)
- Non-PyTorch library calls inside the compiled region

**Debug graph breaks:**
```python
torch.compiler.set_stance("fail_on_recompile")  # raises error on break
# or
torch._dynamo.explain(model, sample_input)      # prints break locations
```

**Fix pattern (replace Python if with torch.where):**
```python
# BAD — causes graph break:
if x.sum() > 0:
    out = f(x)
else:
    out = g(x)

# GOOD — stays in graph:
mask = x.sum(dim=1, keepdim=True) > 0
out = torch.where(mask, f(x), g(x))
```

**Status:** This project's `forward()` is fully graph-friendly. The masking uses pure-tensor
ops (`torch.argsort`, `torch.gather`, `torch.cat`). No graph breaks observed.

---

## What Is NOT Worth Chasing

**Book + NCU evidence:**

| Issue | Why not fixable |
|---|---|
| GEMM 8.33% SM occupancy | Register-bound (222 regs/thread) — Ampere bf16 tensor-core fundamental; no config change fixes it |
| LayerNorm bwd dγ/dβ at 4.5% SM | Cross-batch reduction inherently sequential at small sequence lengths |
| FusedAdamW low-SM invocations | Model has small param tensors (bias, embedding rows); unavoidable |
| 50% of kernels at SM < 10% | `embed_dim=256` is too small to fill 82 SMs; fundamental ceiling |
| Activation checkpointing | Model is 5.17M params; fits 24GB with enormous headroom — no benefit |
| CPU offloading (FSDP/ZeRO) | Single-GPU; not applicable |

---

## Recommended Action Order

```
1. [~2 min]  Switch to max-autotune → -6.3% GPU time (measured)
2. [~15 min] Remove clip_grad_norm_ + timed CUDA graphs experiment → measure samples/sec
3. [optional] Set PYTORCH_ALLOC_CONF=backend:cudaMallocAsync if VRAM growth is a problem
4. [ongoing]  Never add .item() inside the training loop; use torch.cuda.Event for timing
```

---

## Summary

The project is already well-optimized: BF16, FlashAttention, FusedAdamW, DALI, compiled
training, loss-sync once per epoch, and kernel-friendly masking. The remaining low-hanging
fruit is switching `torch.compile` to `max-autotune` (proven -6.3% GPU time in run 16),
removing the grad-clipping sync, and running a proper wall-clock comparison of CUDA graphs
vs. the current larger-batch configuration to settle that open tradeoff.

The fundamental ceiling — 28.6% average SM occupancy on an 82-SM GPU — is not a bug.
It is the consequence of training a 5M-parameter model on a 24GB GPU: the model is simply
too small to fill the hardware. The only way past it is a larger model.
