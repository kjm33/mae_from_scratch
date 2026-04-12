# CPU Usage Analysis: Single-Core Spike During Training

**Date:** 2026-04-12
**Hardware:** AMD Ryzen 9 5950X (16C/32T), RTX 3090 (24GB)
**Observation:** Only one CPU core spikes during training despite having 32 threads available.

---

## 1. Why Only One Core Is Busy

The training loop is thoroughly GPU-bound. The single active core is the Python main thread, which is constrained by the GIL (Global Interpreter Lock). Here's the per-step CPU breakdown:

| CPU Work | Time | Cores Used | State |
|----------|------|------------|-------|
| Dispatch forward CUDA graph | 2.8 ms | 1 (main thread) | Active |
| Wait for GPU (blocked in cudaLaunchKernel) | 190 ms | 1 (main thread) | **Sleeping** |
| Dispatch backward CUDA graph | 2.6 ms | 1 (main thread) | Active |
| Optimizer kernel launches (AdamW8bit) | 24.8 ms | 1 (main thread) | Active |
| DataLoader `__getitem__` | ~0 ms | 6 workers | Idle 99.99% of time |

The spike corresponds to the optimizer phase (252 rapid `cudaLaunchKernel` calls). For the remaining ~195 ms of the step, the core is **blocked/sleeping** waiting on the GPU — it is not spinning or consuming resources. The other 31 threads have nothing to do.

---

## 2. Why DataLoader Workers Are Idle

The dataset is pre-loaded into shared memory. The `__getitem__` method does almost zero work:

```python
def __getitem__(self, idx):
    return self.data[idx].unsqueeze(0)  # ~1 microsecond
```

The 6 worker processes wake up, grab a slice from shared memory, unsqueeze it, and go back to sleep. There is no image decoding, no augmentation, no transforms. This design eliminates data loading as a bottleneck, but it also means the workers have no work to parallelize.

**Current DataLoader config:**
- `num_workers=6`
- `pin_memory=True`
- `persistent_workers=True`
- `prefetch_factor=4`

All correct settings, but overkill given the trivial `__getitem__`. Even `num_workers=2` would produce the same throughput.

---

## 3. The GIL Constraint

Python's Global Interpreter Lock prevents multiple threads from running Python bytecode simultaneously. All PyTorch training orchestration (dispatching CUDA graphs, launching kernels, running the optimizer) happens on the main thread. This is a fundamental constraint of Python-based training, not a misconfiguration.

Key points:
- CUDA graph launches are single API calls — parallelizing them is not possible or beneficial
- The optimizer's 252 sequential kernel launches are serialized by both the GIL and CUDA stream ordering
- `torch.compile` already minimizes CPU-side overhead by fusing operations into graphs

---

## 4. What Would Use More Cores

The only way to give the CPU meaningful parallel work is **data augmentations** running in DataLoader worker processes. Workers run in separate processes (not threads), so they bypass the GIL entirely and use separate cores.

### Augmentations Suitable for MAE Pretraining on OCR Images

```python
import torchvision.transforms.v2 as T

self.transform = T.Compose([
    T.RandomResizedCrop((32, 512), scale=(0.5, 1.0)),  # Standard for MAE, most impactful
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.RandomAutocontrast(p=0.2),
])
```

Each of the 6 workers would run transforms in parallel on different cores while the GPU processes the current batch. This is the standard CPU-GPU overlap pattern.

### Augmentations to Be Careful With (OCR-Specific)

- **`RandomHorizontalFlip`** — flips text horizontally, producing mirrored characters. Bad for downstream OCR unless the encoder is meant to be flip-invariant. Yiddish is RTL, but flipping still distorts individual glyphs.
- **`RandomRotation`** — text lines should stay roughly horizontal for OCR. Small angles (< 3 degrees) might be acceptable as a robustness augmentation.
- **`ColorJitter`** — images are grayscale, so only brightness/contrast apply. Mild values could simulate scan quality variation.

### Impact on CPU Utilization

With `RandomResizedCrop` + 2-3 additional transforms:
- Each `__getitem__` call goes from ~1 us to ~100-500 us
- 6 workers preparing the next batch would use 6 cores in parallel
- Total CPU work per batch: ~25-125 ms (256 images / 6 workers * 100-500 us)
- This would overlap with the 222 ms GPU compute, effectively hiding CPU cost

---

## 5. Other Multi-Core Considerations

### torch.set_num_threads()

Controls the number of threads used by PyTorch's CPU-side operations (intra-op parallelism via OpenMP/MKL). Default is usually the number of physical cores.

```python
torch.set_num_threads(4)  # Reduce from default if workers compete for cores
torch.set_num_interop_threads(2)
```

In this workload, CPU tensor ops are negligible (only `unsqueeze` and `.to(device)`), so this setting has no measurable impact. It becomes relevant if you run CPU-heavy preprocessing in the main thread.

### CUDA Streams for Async Operations

PyTorch already uses a default CUDA stream. Custom streams can overlap data transfers with compute, but `pin_memory=True` + `non_blocking=True` already handles this. The profiler shows only 8 us of GPU memcpy — there's nothing to overlap.

---

## 6. Conclusion

The single-core CPU usage pattern is **normal and expected** for a well-optimized GPU training pipeline with an in-memory dataset. The CPU's role is limited to:

1. Dispatching pre-compiled CUDA graphs (~5 ms)
2. Launching optimizer kernels (~25 ms)
3. Sleeping while the GPU works (~195 ms)

The Ryzen 9 5950X is significantly overprovisioned for this workload. Additional cores can only be utilized by adding data augmentations — which is recommended for model quality (especially `RandomResizedCrop` for MAE) and would have the side benefit of utilizing the idle CPU capacity.

**The bottleneck is and should remain the GPU.** A training pipeline where the CPU is the bottleneck would indicate a data loading or preprocessing problem.
