```torch.cuda.memory_summary(device=accelerator.device)```
This snapshot was taken right after accelerator.prepare() — just the model loaded, no forward pass yet.

  Top-level numbers:

  ┌───────────────────────┬─────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │        Metric         │  Value  │                                              What it means                                              │
  ├───────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Allocated memory      │ ~435 MB │ Tensors actively in use (model weights)                                                                 │
  ├───────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Requested memory      │ ~434 MB │ What your code asked for (slightly less due to alignment padding)                                       │
  ├───────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ GPU reserved memory   │ ~472 MB │ What PyTorch grabbed from CUDA as its memory pool — it over-reserves to avoid repeated cudaMalloc calls │
  ├───────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Non-releasable memory │ ~47 MB  │ Reserved but not currently allocated — padding/fragmentation in the pool                                │
  └───────────────────────┴─────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Large pool vs small pool:

  - Large pool (~425 MB allocated): big tensors — the actual model weight matrices
  - Small pool (~10 MB allocated): small tensors — scalars, bias terms, layer norms, etc.

  Allocations: 254 tensors, 0 freed

  Everything allocated is still live, which makes sense — these are model parameters that persist for the full training run.

  Key takeaway:

  ~435 MB is the model's footprint in float32 at this point. Once training starts, this will grow significantly because:
  - AdamW adds ~2x the parameter memory (momentum + variance buffers) → another ~870 MB
  - Forward pass activations for batch_size=256 add more on top
  - bf16 casting will halve the activation memory but optimizer states stay in fp32

  The ~37 MB gap between reserved (472 MB) and allocated (435 MB) is PyTorch's allocator keeping a buffer to avoid round-tripping to CUDA for small future allocations.
  
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      | 435453 KiB | 435453 KiB | 435453 KiB |      0 B   |
|       from large pool | 424960 KiB | 424960 KiB | 424960 KiB |      0 B   |
|       from small pool |  10493 KiB |  10493 KiB |  10493 KiB |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         | 435453 KiB | 435453 KiB | 435453 KiB |      0 B   |
|       from large pool | 424960 KiB | 424960 KiB | 424960 KiB |      0 B   |
|       from small pool |  10493 KiB |  10493 KiB |  10493 KiB |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      | 433917 KiB | 433917 KiB | 433917 KiB |      0 B   |
|       from large pool | 423424 KiB | 423424 KiB | 423424 KiB |      0 B   |
|       from small pool |  10493 KiB |  10493 KiB |  10493 KiB |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   | 483328 KiB | 483328 KiB | 483328 KiB |      0 B   |
|       from large pool | 471040 KiB | 471040 KiB | 471040 KiB |      0 B   |
|       from small pool |  12288 KiB |  12288 KiB |  12288 KiB |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |  47874 KiB |  57557 KiB | 308030 KiB | 260155 KiB |
|       from large pool |  46080 KiB |  57344 KiB | 300544 KiB | 254464 KiB |
|       from small pool |   1794 KiB |   1856 KiB |   7486 KiB |   5691 KiB |
|---------------------------------------------------------------------------|
| Allocations           |     254    |     254    |     254    |       0    |
|       from large pool |      73    |      73    |      73    |       0    |
|       from small pool |     181    |     181    |     181    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |     254    |     254    |     254    |       0    |
|       from large pool |      73    |      73    |      73    |       0    |
|       from small pool |     181    |     181    |     181    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |      29    |      29    |      29    |       0    |
|       from large pool |      23    |      23    |      23    |       0    |
|       from small pool |       6    |       6    |       6    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |      24    |      24    |      29    |       5    |
|       from large pool |      21    |      21    |      23    |       2    |
|       from small pool |       3    |       3    |       6    |       3    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|