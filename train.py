import os
import time

import cv2
import torch
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from mae import MaskedAutoencoderViT, YiddishSharedInRamDataset

_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir

IMG_SIZE = (32, 512)
LOG_DIR = "runs/mae_yiddish"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
# Prefer this file for TensorBoard reconstruction when present in lines_dir
PREFERRED_MONITOR_IMAGE = "BN_523.715_0013.tsv.processed_LINE_5.TIF"

TENSORBOARD_PROFILE = "1_inital_reference_version"


def find_monitor_image(lines_dir):
    """Return path to monitor image: preferred file if present, else first image in lines_dir."""
    if not os.path.isdir(lines_dir):
        return None
    preferred_path = os.path.join(lines_dir, PREFERRED_MONITOR_IMAGE)
    if os.path.isfile(preferred_path):
        return preferred_path
    names = [
        f for f in os.listdir(lines_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not names:
        return None
    names.sort()
    return os.path.join(lines_dir, names[0])


def load_monitor_image(path, img_size, device):
    """Load one image for TensorBoard reconstruction logging; resize to img_size (H, W)."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return x.to(device)


def log_reconstruction(writer, model, monitor_img, epoch, mask_ratio=0.75):
    """Run model on monitor image and log original + reconstructed image to TensorBoard."""
    m = model.module if hasattr(model, "module") else model
    model.eval()
    with torch.no_grad():
        _, pred, _ = model(monitor_img, mask_ratio=mask_ratio)
        recon = m.unpatchify(pred)  # (1, 1, H, W)
    model.train()
    # Preds may be in normalized space when norm_pix_loss=True; scale to [0,1] for display
    r = recon[0].cpu().float()
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    orig = monitor_img[0].cpu().float()
    writer.add_image("monitor/original", orig, epoch, dataformats="CHW")
    writer.add_image("monitor/reconstructed", r, epoch, dataformats="CHW")


def train():
    accelerator = Accelerator(mixed_precision="bf16")

    model = MaskedAutoencoderViT(
        img_size=(32, 512),
        patch_size=8,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        norm_pix_loss=True,
    )

    lines_dir = "./data/yiddish_lines"
    dataset = YiddishSharedInRamDataset(lines_dir, img_size=(32, 512))
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # model = model.to(accelerator.device) # accelerator already loads the model into GPU
    model = torch.compile(model, mode="reduce-overhead")

    model.train()

    prof = None
    if accelerator.is_local_main_process:
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=tensorboard_trace_handler(f"./runs/{TENSORBOARD_PROFILE}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    for epoch in range(10):
        for step, batch in enumerate(dataloader):
            batch = batch.to(accelerator.device, non_blocking=True)
            batch = batch.to(torch.bfloat16).div_(255.0)

            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(batch, mask_ratio=0.75) # high CPU usage - due to torch.compilation

            accelerator.backward(loss) # CPU peak
            optimizer.step()

            if prof is not None:
                prof.step()
                if step >= 4:  # wait(1) + warmup(1) + active(3) = 5 steps
                    prof.stop()
                    prof = None


if __name__ == "__main__":
    train()
