import os

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')


def build_dali_loader(root_dir, img_size=(32, 512), batch_size=256, num_threads=4, device_id=0):
    """Build a DALI data pipeline for grayscale text-line images.

    Reads images from disk, GPU-decodes them, resizes to img_size, and normalizes
    to [0, 1] float32. Returns a DALIGenericIterator that yields dicts with key
    "images" containing tensors of shape (N, 1, H, W) already on GPU.

    Args:
        root_dir:    Directory containing image files.
        img_size:    Target (H, W). Default (32, 512).
        batch_size:  Samples per batch.
        num_threads: CPU threads for prefetch/decode.
        device_id:   GPU index.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Path does not exist: {os.path.abspath(root_dir)}")

    file_list = sorted(
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    )
    if not file_list:
        raise ValueError(
            f"No images found in: {os.path.abspath(root_dir)}. "
            f"Supported extensions: {VALID_EXTENSIONS}"
        )

    print(f"DALI pipeline: {len(file_list)} images from {root_dir}")

    h, w = img_size

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def _pipeline():
        encoded, _ = fn.readers.file(
            files=file_list,
            shuffle_after_epoch=True,
            name="Reader",
        )
        # Mixed device: CPU decode start -> GPU decode finish (faster than pure CPU)
        images = fn.decoders.image(encoded, device="mixed", output_type=types.GRAY)
        images = fn.resize(images, device="gpu", resize_y=h, resize_x=w,
                           interp_type=types.INTERP_LINEAR)
        # Normalize to [0, 1], convert HWC -> CHW, cast to float32
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
    )
