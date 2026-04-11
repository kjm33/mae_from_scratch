import os
import time

import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import profiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
import bitsandbytes as bnb
from training_logger import TrainingLogger

from mae import MaskedAutoencoderViT, YiddishSharedInRamDataset

_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir

IMG_SIZE = (32, 512)
LOG_DIR = "runs/mae_yiddish"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
# Prefer this file for TensorBoard reconstruction when present in lines_dir
PREFERRED_MONITOR_IMAGE = "BN_523.715_0013.tsv.processed_LINE_5.TIF"

TENSORBOARD_PROFILE = "4_single_step"


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
    device = torch.device("cuda")

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
    ).to(device)

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

    #from https://medium.com/@jiminlee-ai/why-your-tiny-deep-learning-model-is-hogging-all-your-gpu-vram-85bc58ee5050
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    model = torch.compile(model, mode="reduce-overhead")

    num_epochs = 6
    model.train()

   # with TrainingLogger(device, num_epochs, len(dataloader), TENSORBOARD_PROFILE) as logger:
    for epoch in range(num_epochs):
        # logger.begin_epoch(epoch)

        for step, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True).float().div_(255.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model(batch, mask_ratio=0.75)

            loss.backward()
            optimizer.step()

            # logger.on_step(loss.item())

        # logger.end_epoch(epoch)

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=tensorboard_trace_handler(f"./runs/{TENSORBOARD_PROFILE}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True,
        ) as prof:
            with profiler.record_function("train_step"):
                torch.cuda.nvtx.range_push("forward")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _, _ = model(batch, mask_ratio=0.75)
                # end of forward
                torch.cuda.nvtx.range_pop()

                # Backward pass and optimization
                torch.cuda.nvtx.range_push("backward")
                loss.backward()

                torch.cuda.nvtx.range_push("optimizer_step")
                optimizer.step()
                # end of optimizer_step

                torch.cuda.nvtx.range_pop()
                optimizer.zero_grad()
                # end of backward
                torch.cuda.nvtx.range_pop()

    

        


if __name__ == "__main__":
    train()
