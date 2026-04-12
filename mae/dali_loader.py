import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def build_dali_loader(npy_path, batch_size=256, num_threads=4, device_id=0):
    """Build a DALI data pipeline backed by a pre-processed numpy memmap file.

    Expects npy_path to be a (N, H, W) uint8 array produced by prepare_dataset.py.
    Returns a DALIGenericIterator yielding dicts with key "images" containing
    tensors of shape (N, 1, H, W) float32 in [0, 1], already on GPU.

    Args:
        npy_path:    Path to the .npy file produced by prepare_dataset.py.
        batch_size:  Samples per batch.
        num_threads: CPU threads for the external source callback.
        device_id:   GPU index.
    """
    data = np.load(npy_path, mmap_mode='r')   # (N, H, W) uint8, memory-mapped
    n = len(data)
    print(f"DALI pipeline: {n} images from {npy_path}  (memmap, no per-epoch decode)")

    # Pre-shuffle index array; DALI external_source handles epoch-level reshuffling
    # by passing a new permutation each time the iterator resets.
    rng = np.random.default_rng()

    def source(sample_info):
        if sample_info.iteration == 0 and sample_info.idx_in_epoch == 0:
            # New epoch — reshuffle
            source._perm = rng.permutation(n)
        idx = source._perm[sample_info.idx_in_epoch + sample_info.iteration * batch_size]
        # Return (H, W, 1) uint8 so DALI treats it as a single-channel HWC image
        return data[idx, :, :, np.newaxis].copy()

    source._perm = rng.permutation(n)

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def _pipeline():
        images = fn.external_source(
            source=source,
            num_outputs=1,
            dtype=types.UINT8,
            layout="HWC",
        )
        images = images.gpu()
        images = fn.crop_mirror_normalize(
            images,
            device="gpu",
            mean=[0.0],
            std=[255.0],
            output_layout="CHW",
            dtype=types.FLOAT,
        )
        return images

    pipe = _pipeline()
    pipe.build()

    return DALIGenericIterator(
        pipe,
        output_map=["images"],
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
        size=n,
    )
