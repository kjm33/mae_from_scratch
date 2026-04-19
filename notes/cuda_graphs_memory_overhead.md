# Why CUDA Graphs Use ~7 GB More VRAM

## Observation

NCU profiling (TORCHDYNAMO_DISABLE=1, eager mode): ~16 GB VRAM
Normal training (torch.compile reduce-overhead, CUDA graphs): ~23 GB VRAM

Difference: ~7 GB at batch_size=5120.

## Cause: Static Memory Allocation

CUDA graphs require fully static memory allocation. When the graph is captured, PyTorch
pre-allocates buffers for every intermediate tensor in the entire train_step simultaneously:

- All encoder activations (needed for backward)
- All decoder activations
- All gradient buffers
- Optimizer state buffers (momentum, variance)
- clip_grad_norm_ workspace

These buffers stay live for the graph's lifetime. The allocator cannot reuse memory between
operations because the graph layout is fixed.

In eager mode, the allocator works dynamically — a tensor that goes out of scope during the
backward pass can have its memory immediately reused for the next allocation. Peak usage is
lower because not everything needs to be alive simultaneously.

The ~7 GB gap is the CUDA graph static allocation overhead at batch_size=5120. It is the
cost of getting near-zero CPU overhead per step (two cudaGraphLaunch calls instead of
hundreds of individual kernel dispatches).
