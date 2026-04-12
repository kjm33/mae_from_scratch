# Profiler Analysis — Run 6 (take_loss_once_per_epoch)

## Summary

Removing `loss.item()` from the hot loop eliminated the dominant CPU bottleneck.
CPU op time for 3 training steps dropped from **2.57s → 132ms (19× improvement)**.

## Before vs After

| Metric | Run 5 | Run 6 |
|--------|-------|-------|
| CPU op time (3 steps) | 2.57 s | 132 ms |
| Per-step sync | 403 ms `cudaStreamSynchronize` | none |
| GPU kernel density | 99.2% | 99.7% |
| Idle gaps > 50µs | 12 (8.8 ms total) | 6 (2.1 ms total) |

## Remaining CPU Work Per Step (~44ms, non-blocking)

GPU runs 407ms per step while CPU does:

```
29.1% | ~12.8ms | Torch-Compiled Region: 0/0   — fwd graph CPU-side management
28.6% | ~12.6ms | Torch-Compiled Region: 1/0   — bwd graph CPU-side management
 9.1% |  ~4.0ms | CompiledFunction
 2.5% |  ~1.1ms | AccumulateGrad ×252           — leaf grad accumulation, 1 per param
 0.9% |  ~1.2ms | TorchDynamo Cache Lookup
```

None of this blocks the GPU. CPU completes in ~44ms while GPU runs 407ms asynchronously.

## AccumulateGrad ×252

Even with `torch.compile`, the autograd engine still runs 252 leaf gradient
accumulation callbacks through Python (one per model parameter) rather than inside
the CUDA graph. At ~1.1ms/step it is not a bottleneck, but it is the last remaining
piece of Python autograd overhead that falls outside the compiled graph.

## Profiler Artifact

The `cudaDeviceSynchronize x1 = 1.20s` in the training steps trace is inserted by
the profiler itself when the active capture window closes — it flushes all GPU events
before writing the trace file. It is one call, not per-step, and does not exist
outside profiling.

## Conclusion

GPU kernel density is 99.7% with no idle gaps > 643µs. The CPU is no longer a
bottleneck. No further CPU/GPU overlap improvements are available at this level.
