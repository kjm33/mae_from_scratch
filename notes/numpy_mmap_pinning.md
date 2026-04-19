# Pinning numpy mmap for CUDA transfers

## What "pinning" means

Normal RAM is *pageable* — the OS can swap pages to disk, and when the GPU DMA engine needs
to read from it, CUDA internally copies it through a *staging bounce buffer* that is
page-locked. Pinned (page-locked) memory skips that bounce: the GPU DMA reads directly from
the host buffer, faster and with lower CPU overhead.

## Current transfer chain in dali_loader.py

```
mmap (OS page cache, pageable)
  → .copy()  at line 42         →  new pageable numpy array
  → DALI external_source        →  DALI internal pinned staging buffer  (copy #2)
  → .gpu()                      →  GPU (DMA from staging)
```

Two copies happen on CPU before the data reaches the GPU.

---

## Option 1 — `cudaHostRegister` the entire mmap

Register the mmap region itself as page-locked so DALI can DMA from it directly:

```python
import ctypes
import numpy as np

cudart = ctypes.CDLL("libcudart.so")

data = np.load(npy_path, mmap_mode='r')

# Fault in all pages first — mmap is lazy, pages don't exist in RAM until touched
_ = data.sum()

# Register as CUDA page-locked memory
ret = cudart.cudaHostRegister(
    ctypes.c_void_p(data.ctypes.data),
    ctypes.c_size_t(data.nbytes),
    ctypes.c_uint(0),  # cudaHostRegisterDefault
)
assert ret == 0, f"cudaHostRegister failed: {ret}"
```

Unregister on cleanup:
```python
cudart.cudaHostUnregister(ctypes.c_void_p(data.ctypes.data))
```

**Downside:** locks the entire dataset in RAM permanently. Page-locked memory is a scarce
system resource — if the dataset is large (several GB), this can starve other processes and
even destabilize the system.

---

## Option 2 — Load into RAM, pin with PyTorch

Simpler and safer if the dataset fits:

```python
data_ram = np.array(np.load(npy_path, mmap_mode='r'))   # copy to regular RAM
data_pinned = torch.from_numpy(data_ram).pin_memory()    # pin it
data = data_pinned.numpy()                               # numpy view, no extra copy
```

`torch.pin_memory()` calls `cudaMallocHost` under the hood — cleaner than `cudaHostRegister`
and PyTorch manages the lifetime.

---

## Option 3 — `madvise` (not true pinning, but helps)

If the dataset is too large to pin entirely, tell the kernel to keep pages in RAM and
prefetch ahead:

```python
import mmap as mmap_mod
data = np.load(npy_path, mmap_mode='r')
mmap_mod.madvise(data.ctypes.data, data.nbytes,
                 mmap_mod.MADV_WILLNEED | mmap_mod.MADV_SEQUENTIAL)
```

Pages aren't locked — the OS can still evict them — but for a hot dataset accessed every
epoch they'll generally stay warm.

---

## Note for this project

Training is at 98.5% GPU kernel density with ~10 ms/step. The DALI pipeline runs on separate
CPU threads and pre-fetches ahead of the GPU. Unless the profiler shows DALI stalls or idle
time before GPU kernels, data loading is not the bottleneck and none of these options will
move step time.
