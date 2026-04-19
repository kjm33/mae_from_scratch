import argparse
import os
import time

# Suppress TensorFlow/absl noise (imported transitively by TensorBoard)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import logging

import cv2
import torch

# torch.compile can't trace profiler record_function annotations — expected, not a bug
logging.getLogger("torch._dynamo.variables.torch").setLevel(logging.ERROR)
from torch.utils.tensorboard import SummaryWriter
from training_logger import TrainingLogger

from mae import MaskedAutoencoderViT, mae_vit_ultra_light
from mae.dali_loader import build_dali_loader

_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir

IMG_SIZE = (32, 512)
LOG_DIR = "runs/mae_yiddish"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
# Prefer this file for TensorBoard reconstruction when present in lines_dir
PREFERRED_MONITOR_IMAGE = "BN_523.715_0013.tsv.processed_LINE_5.TIF"



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


def train(profile: str | None = None):
    device = torch.device("cuda")

    model = mae_vit_ultra_light().to(device)

    dataloader = build_dali_loader("./data/yiddish_lines.npy", batch_size=256, num_threads=4)

    # lr=1.5e-4 is tuned for batch_size=256. Linear scaling rule: lr = 1.5e-4 * (batch_size / 256).
    # At batch_size=4096 that gives 2.4e-3. MAE is tolerant of deviations but worth aligning
    # for real training runs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=0.05, fused=True)

    num_epochs = 6
    # Cosine decay with 5% linear warmup — matches the MAE paper's schedule.
    # max_lr=3.0e-3 is the linearly scaled lr for batch_size=5120 (base 1.5e-4 at 256).
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3.0e-3,
        steps_per_epoch=len(dataloader),
        epochs=num_epochs,
        pct_start=0.05,
        anneal_strategy="cos",
    )

    # Compile the full training step (forward + backward + optimizer) into a single
    # CUDA graph so the optimizer kernel launches are captured too, eliminating
    # per-parameter dispatch overhead. max-autotune benchmarks multiple kernel
    # implementations (GEMM tilings, Triton configs) and picks the fastest —
    # longer first-step compile, but results are cached in .cache/.
    @torch.compile(mode="max-autotune")
    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("forward"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model(batch, mask_ratio=0.75)
        with torch.profiler.record_function("backward"):
            loss.backward()
        with torch.profiler.record_function("optimizer_step"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        return loss

    num_epochs = 6
    model.train()

    with TrainingLogger(device, num_epochs, len(dataloader), profile) as logger:
        for epoch in range(num_epochs):
            logger.begin_epoch(epoch)
            epoch_loss = torch.zeros(1, device=device)

            for step, batch_data in enumerate(dataloader):
                batch = batch_data[0]["images"]  # (N, 1, H, W) float32 on GPU
                loss = train_step(batch)
                scheduler.step()  # CPU-side lr update — outside compiled graph
                epoch_loss += loss.detach()
                logger.on_step()

            avg_loss = (epoch_loss / len(dataloader)).item()  # one sync per epoch
            logger.end_epoch(epoch, avg_loss)

        if profile:
            with logger.profile_step():
                train_step(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="Profile name — captures a trace to runs/<name>_<timestamp>.pt.trace.json")
    args = parser.parse_args()
    train(profile=args.profile)
