# Are CUDA Graphs Worth It for This Project?

## Conclusion: Probably not at batch_size=5120+

## Why CUDA graphs help less at large batch sizes

CUDA graphs eliminate CPU dispatch overhead — instead of launching hundreds of kernels
individually, just two cudaGraphLaunch calls. This matters most when:
- Each GPU kernel finishes quickly (small model, small batch)
- The CPU scrambles to dispatch the next kernel before the GPU is idle
- CPU→GPU dispatch gap is a measurable fraction of step time

At batch_size=5120, each GEMM operates on large tensors (5120×256, 5120×64×256, etc.).
Each kernel takes long enough that the CPU has time to dispatch the next one before the GPU
finishes. Dispatch overhead is already hidden by long-running kernels.

## Practical tradeoff

| | With graphs (reduce-overhead) | Without graphs (default) |
|---|---|---|
| VRAM | ~23 GB | ~16 GB |
| Max batch on RTX 3090 | ~5120 | ~7168 |
| CPU dispatch overhead | ~0 | small, mostly hidden |
| Kernel fusion (Triton) | yes | yes |

## Recommendation

Switch to `mode="default"` — keeps all Triton kernel fusion but drops CUDA graphs,
freeing ~7 GB. Use the freed memory for a larger batch instead.

For MAE pretraining, larger batches give more diverse masked views per update, which is
likely a bigger win than eliminating already-small dispatch overhead.

```python
@torch.compile(mode="default")
def train_step(batch):
    ...
```

Verify: if step time is within a few percent of reduce-overhead, keep default and push
batch_size to 7168.
