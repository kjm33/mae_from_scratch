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
