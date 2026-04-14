# Dataset Loading Format Comparison

## Problem

Loading thousands of small image files per epoch has high overhead:
- One `open()` + `read()` syscall per file per epoch
- No sequential prefetch benefit
- High inode/metadata lookup cost

## Options

| Format | DALI reader | Decode/resize per epoch | Random access | Notes |
|--------|-------------|------------------------|---------------|-------|
| Individual files | `fn.readers.file` | Yes | O(1) per file, but many syscalls | Current approach |
| WebDataset (tar) | `fn.readers.webdataset` | Yes | Sequential only | Native DALI; images stay encoded, smaller on disk |
| MXNet RecordIO | `fn.readers.mxnet` | Yes | Sequential | Native DALI binary pack |
| **numpy memmap** | `fn.external_source` | **No** | O(1), OS-managed | Pre-resized uint8; decode/resize cost paid once |

## Chosen: numpy memmap

Single `.npy` file storing all images pre-resized to (N, 32, 512) uint8.

**Pros:**
- Decode and resize happen once (preparation script), not every epoch
- OS memory-maps the file — hot pages stay in RAM, cold pages streamed in
- Random index access is a direct byte offset: `offset = idx * H * W`
- File size equals current RAM usage (N × 32 × 512 bytes ≈ same as in-RAM tensor)

**Cons:**
- Larger on disk than encoded formats (WebDataset stores compressed images)
- Preparation step required before first training run

## Implementation

- `prepare_dataset.py` — one-time script: decode + resize all images → save as `.npy`
- `mae/dali_loader.py` — `build_dali_loader` uses `fn.external_source` with a numpy memmap

## Sequential vs Random Access

For training with shuffled batches, random access is required. numpy memmap gives O(1)
random access (direct byte offset), unlike tar-based formats which are sequential-only
and would require buffered shuffling.

## Why Not GPU JPEG Decoding (DALI + nvJPEG)?

DALI supports `fn.decoders.image(device="mixed")` which decodes JPEGs on the GPU via
nvJPEG. This keeps images compressed on disk (smaller storage) but loses on every other
dimension for this dataset:

1. **nvJPEG overhead is per-image, not per-pixel.** For megapixel photos the decode work
   dominates the launch cost. For 32×512 thumbnails (~16KB uncompressed), the per-image
   kernel launch is a large fraction of total work — many tiny GPU kernels per batch
   instead of a few large matmuls.

2. **The cost repeats every epoch.** Decode happens N_images × N_epochs times. memmap
   pays it once in `prepare_dataset.py`. At 800 epochs × 100K images that's 80M decode
   operations eliminated.

3. **GPU decode competes with training kernels.** nvJPEG runs on the GPU's copy engines
   and compute units. With the memmap pipeline at 99.7% kernel density the GPU is already
   saturated; adding decode would stall training kernels or require complex stream
   scheduling.

4. **uint8 → float32 cast is nearly free.** `fn.crop_mirror_normalize` is a fused
   scale+offset kernel over contiguous memory — bandwidth-bound, not compute-bound like
   JPEG entropy decoding.

**The one real win for GPU decode:** disk space. Encoded JPEGs for 32×512 grayscale are
~2–5KB vs 16KB raw uint8 (3–8× smaller). For this dataset size that's unlikely to matter,
and the savings are paid back in GPU time every epoch.
