"""One-time script: decode + resize all images and pack into a single numpy memmap file.

Usage:
    python prepare_dataset.py

Output:
    ./data/yiddish_lines.npy  — shape (N, 32, 512) uint8, C-contiguous

Re-run whenever the image directory changes.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_DIR = "./data/yiddish_lines"
OUT_PATH = "./data/yiddish_lines.npy"
IMG_SIZE = (32, 512)          # (H, W)
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')


def main():
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"Source directory not found: {os.path.abspath(SRC_DIR)}")

    names = sorted(
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith(VALID_EXTENSIONS)
    )
    if not names:
        raise ValueError(f"No images found in {SRC_DIR}")

    n = len(names)
    h, w = IMG_SIZE
    print(f"Preparing {n} images -> {OUT_PATH}  ({n * h * w / 1e6:.1f} MB)")

    # Write directly into a memmap so we never hold the full dataset in Python memory
    out = np.lib.format.open_memmap(OUT_PATH, mode='w+', dtype=np.uint8, shape=(n, h, w))

    errors = 0
    for idx, name in enumerate(tqdm(names)):
        path = os.path.join(SRC_DIR, name)
        try:
            with Image.open(path) as img:
                img = img.convert('L').resize((w, h), resample=Image.BILINEAR)
                out[idx] = np.asarray(img, dtype=np.uint8)
        except Exception as e:
            print(f"\nError loading {name}: {e} — filling with zeros")
            out[idx] = 0
            errors += 1

    out.flush()
    print(f"Done. Shape: {out.shape}, dtype: {out.dtype}, errors: {errors}")
    print(f"Saved to: {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
