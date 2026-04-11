# Observations
during trainig CPU usage is 100% on a single core

# accelerator disabled
first epoch took 79.7s comparing to 17.2s with accelerator
^^ rebuilding torch.compile cache

● Here's a full line-by-line resource map:

  LINE(S)   | VRAM          | RAM           | CPU
  ----------|---------------|---------------|------------------------------------------
  70        | ~500 MB       | moderate      | CUDA context init
            | (CUDA ctx)    |               |
  ----------|---------------|---------------|------------------------------------------
  72–82     | 0             | ~435 MB       | builds model graph, initializes weights
            |               | (fp32 params) |
  ----------|---------------|---------------|------------------------------------------
  85        | 0             | N × 16KB      | loads & resizes every image,
            |               | (all images)  | single-threaded, tqdm loop
            |               | + shared mem  |
  ----------|---------------|---------------|------------------------------------------
  86–94     | 0             | 6 workers     | forks 6 worker processes,
            |               | × ~100 MB     | each copies Python runtime
            |               | + prefetch    | (persistent, stay alive forever)
            |               | buffers       |
  ----------|---------------|---------------|------------------------------------------
  96        | 0             | ~870 MB       | allocates momentum + variance
            |               | (2× params,   | buffers for all params
            |               | fp32)         |
  ----------|---------------|---------------|------------------------------------------
  98        | ~435 MB       | freed from    | DDP wrapping, moves model to GPU,
            | (model fp32   | CPU (model    | registers gradient hooks on all params
            | → bf16 later) | moves to GPU) |
  ----------|---------------|---------------|------------------------------------------
  100       | 0             | moderate      | deferred — no work yet,
            |               |               | compilation triggers on first call
  ----------|---------------|---------------|------------------------------------------
  105–106   | +8 MB         | worker reads  | worker processes decode batch from
            | (batch on GPU)| shared mem,   | shared RAM, pin_memory copies to
            |               | pin buffers   | page-locked memory, DMA to GPU
  ----------|---------------|---------------|------------------------------------------
  107       | 0 (in-place)  | 0             | negligible — cast + scale in-place on GPU
  ----------|---------------|---------------|------------------------------------------
  109       | frees grad    | 0             | negligible — sets grad ptrs to None
            | memory        |               |
  ----------|---------------|---------------|------------------------------------------
  110       | peak usage:   | moderate      | HIGH on step 0: torch.compile traces +
            | params +      | (autograd     | compiles Triton kernels (~2–5 min)
            | activations   | graph)        | subsequent steps: autograd graph build
            | + intermediates               |
  ----------|---------------|---------------|------------------------------------------
  113       | grad tensors  | 0             | HIGH: autograd graph traversal +
            | allocated     |               | DDP all-reduce hooks fire per bucket
            | per param     |               | (NCCL launches on CPU)
  ----------|---------------|---------------|------------------------------------------
  114       | grad memory   | 0             | HIGH: Python loop dispatches one CUDA
            | freed after   |               | kernel per param tensor (AdamW update)
            | set_to_none   |               |

  Biggest bottlenecks ranked:

  ┌──────────┬───────┬───────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
  │ Priority │ Line  │                 Issue                 │                               Fix                               │
  ├──────────┼───────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ 1        │ 85    │ All images loaded into RAM            │ already optimal with share_memory_() — just a dataset size cost │
  ├──────────┼───────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ 2        │ 86–94 │ 6 persistent workers × Python runtime │ reduce num_workers if RAM is tight                              │
  ├──────────┼───────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ 3        │ 114   │ AdamW Python dispatch loop            │ use fused=True on the optimizer                                 │
  ├──────────┼───────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ 4        │ 113   │ DDP all-reduce hooks per bucket       │ increase bucket size                                            │
  ├──────────┼───────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ 5        │ 96    │ Optimizer state ~870 MB in RAM        │ states move to GPU after prepare() — RAM freed                  │
  └──────────┴───────┴───────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘

  The RAM you saw (~4 GB) is mostly lines 85 (images) + 86–94 (workers) + 96 (optimizer states before they move to GPU).

✻ Brewed for 31s
