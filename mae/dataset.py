import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class YiddishSharedInRamDataset(Dataset):
    """Loads all grayscale text-line images into shared memory for zero-copy DataLoader access."""

    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')

    def __init__(self, root_dir, img_size=(32, 512)):
        self.root_dir = root_dir
        self.img_size = img_size

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Path does not exist: {os.path.abspath(root_dir)}")

        self.file_names = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(self.VALID_EXTENSIONS)
        ]

        if len(self.file_names) == 0:
            raise ValueError(
                f"No images found in: {os.path.abspath(root_dir)}. "
                f"Supported extensions: {self.VALID_EXTENSIONS}"
            )

        print(f"Loading {len(self.file_names)} images into RAM...")

        # Pre-allocate tensor for all images
        self.data = torch.empty((len(self.file_names), img_size[0], img_size[1]), dtype=torch.uint8)

        for idx, name in enumerate(tqdm(self.file_names)):
            img_path = os.path.join(self.root_dir, name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L').resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
                    self.data[idx] = torch.from_numpy(np.array(img, dtype=np.uint8))
            except Exception as e:
                print(f"Error loading {name}: {e}")
                self.data[idx] = torch.zeros((img_size[0], img_size[1]), dtype=torch.uint8)

        self.data = self.data.share_memory_()
        print(f"Dataset ready. Loaded: {self.data.shape[0]} images.")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0)
