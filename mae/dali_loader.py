import warnings

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def build_dali_loader(npy_path, batch_size=256, num_threads=4, device_id=0,
                      shard_id=0, num_shards=1):
    """Build a DALI data pipeline backed by a pre-processed numpy memmap file.

    Expects npy_path to be a (N, H, W) uint8 array produced by prepare_dataset.py.
    Returns a DALIGenericIterator yielding dicts with key "images" containing
    tensors of shape (N, 1, H, W) float32 in [0, 1], already on GPU.

    Args:
        npy_path:    Path to the .npy file produced by prepare_dataset.py.
        batch_size:  Samples per batch (per GPU).
        num_threads: CPU threads for the external source callback.
        device_id:   GPU index.
        shard_id:    Index of this GPU's data shard (0-based).
        num_shards:  Total number of shards (= number of GPUs).
    """
    data = np.load(npy_path, mmap_mode='r')   # (N, H, W) uint8, memory-mapped
    n = len(data)

    # Divide dataset into equal shards; each GPU only sees its own shard.
    shard_size = n // num_shards
    shard_indices = np.arange(shard_id * shard_size, (shard_id + 1) * shard_size)
    print(f"DALI pipeline [{shard_id}/{num_shards}]: {shard_size} images from {npy_path}")

    rng = np.random.default_rng()

    # Number of complete batches per epoch within this shard (DROP policy)
    num_batches = shard_size // batch_size
    epoch_size = num_batches * batch_size

    # batch=True: DALI calls source(epoch_idx) once per batch; epoch_idx is a plain int.
    # Track iteration manually since DALI does not pass it in this mode.
    state = {"epoch": -1, "iteration": 0, "perm": rng.permutation(shard_indices)}

    def source(epoch_idx):
        if epoch_idx != state["epoch"]:
            state["epoch"] = epoch_idx
            state["iteration"] = 0
            state["perm"][:] = rng.permutation(shard_indices)
        start = state["iteration"] * batch_size
        state["iteration"] += 1
        idx = state["perm"][start:start + batch_size]
        # data[idx] is (B, H, W) uint8; add channel dim → list of (H, W, 1) arrays
        batch = data[idx]  # one vectorised memmap read
        return [batch[i, :, :, np.newaxis].copy() for i in range(batch_size)]

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                  prefetch_queue_depth=1)
    def _pipeline():
        images = fn.external_source(
            source=source,
            batch=True,
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

    # reader_name is the new API but only applies to file readers, not external_source.
    # size= is the only way to give DALIGenericIterator a correct len(); suppress the
    # inapplicable deprecation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Please set `reader_name`")
        return DALIGenericIterator(
            pipe,
            output_map=["images"],
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
            size=epoch_size,  # per-shard epoch size
        )
