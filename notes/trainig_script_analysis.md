# Batch size

 - Shape: (256, 1, 32, 512) — batch_size × channels × H × W
  - bf16 = 2 bytes per element
  - 256 × 1 × 32 × 512 × 2 = 8,388,608 bytes = **8 MB**

# RAM usage


  1. The dataset itself (biggest likely culprit)

  Line 35 pre-allocates all images as uint8 tensors:
  N images × 32 × 512 × 1 byte = N × 16,384 bytes
  For example:
  - 10,000 images → ~156 MB
  - 100,000 images → ~1.6 GB
  - 250,000 images → ~4 GB

  How many images do you have?
  ls data/yiddish_lines/ | wc -l

  2. num_workers=6 with persistent_workers=True

  Each worker is a forked Python process. Even though the dataset uses share_memory_() (line 47) to avoid copying the image data, each worker still carries:
  - Python interpreter overhead (~50–100 MB each)
  - PyTorch runtime, imports, etc.
  - 6 workers × ~100 MB = ~600 MB

  3. prefetch_factor=4

  Each worker prefetches 4 batches into CPU memory. With 6 workers:
  - Each batch: 256 × 1 × 32 × 512 = 4 MB (uint8)
  - 6 workers × 4 prefetched batches × 4 MB = ~96 MB

  4. CUDA context + model weights in CPU

  PyTorch's CUDA context itself takes ~500 MB of virtual memory, and the model parameters briefly live in CPU RAM before being moved to GPU.

  ---
  The dataset size is almost certainly the dominant factor. Check your image count and you'll likely find that accounts for most of the 4 GB.

# Optimizer
optimizer.step() <- + 2 GB of VRAM


● Line 113 is optimizer.step(). The CPU spike comes from how AdamW dispatches work to the GPU.

  Root cause: Python loop over parameter groups

  PyTorch's AdamW iterates over every parameter tensor in Python to launch a separate CUDA kernel for each one. Your ViT-Base has many parameter tensors (embeddings, 12 encoder blocks + 8 decoder blocks,
  norms, etc.), so this is a tight Python loop launching hundreds of small GPU kernels — pure CPU overhead.

  Why it's visible as a spike

  GPU operations are async — the CPU queues them and moves on. But optimizer.step() forces the CPU to briefly synchronize to ensure gradients are ready, then burns CPU time in that Python dispatch loop.

  Fix: use fused or foreach AdamW

   # fused: single CUDA kernel for all params — lowest CPU overhead (requires CUDA)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, fused=True)

   # foreach: batches the per-tensor ops — also much better than default
  optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, foreach=True)

  fused=True is the best option here — it collapses all parameter updates into a single kernel launch, eliminating the Python loop entirely. It's compatible with bf16 and your RTX 3090s support it.

  Note that torch.compile partially mitigates this by fusing optimizer steps into the graph, but fused=True handles it more directly.and why does 

 # accelerator.backward(loss) is essentially loss.backward() CPU spike

   1. Autograd engine graph traversal

  PyTorch's autograd engine runs on CPU. During the forward pass it built a computation graph; backward() traverses that graph node-by-node on CPU to schedule gradient kernels on the GPU. This traversal is
  pure Python/C++ CPU work — the more ops in the graph (12 encoder blocks + 8 decoder blocks = a lot of nodes), the more CPU time spent scheduling.

  2. DDP gradient bucketing hooks

  Since you're running 2-GPU DDP via Accelerate, DDP registers backward hooks on every parameter. As each bucket of gradients becomes ready during backward, the hook fires on CPU to launch an all-reduce (NCCL)
   operation to sync gradients across both GPUs. These hooks firing repeatedly throughout the backward pass is visible as sustained CPU activity.

  The sequence looks like:
  backward starts
    → GPU computes gradients for a layer
    → CPU hook fires → launches NCCL all-reduce for that bucket
    → GPU computes next layer's gradients
    → CPU hook fires → launches NCCL all-reduce ...
    → repeat for every bucket

  How to reduce it

  Increase the DDP gradient bucket size to reduce the number of all-reduce calls:
  accelerator = Accelerator(
      mixed_precision="bf16",
      gradient_accumulation_steps=1,
  )
  # Set larger bucket size (default is 25MB)
  torch.distributed.broadcast_object_list  # not directly via Accelerate

  Or set it directly before prepare():
  from torch.nn.parallel import DistributedDataParallel
  # After prepare, set bucket cap:
  model._orig_mod.bucket_bytes_cap = 50 * 1024 * 1024  # 50MB

  Though with torch.compile, the graph is partially fused so this is less impactful than in eager mode.