import argparse
import os
import time

# Suppress TensorFlow/absl noise (imported transitively by TensorBoard)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import logging

import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
CHECKPOINT_PATH = "checkpoints/checkpoint.pt"
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


class _TrainStep(torch.nn.Module):
    """Wraps the three MAE sub-functions into one forward() so DDP can hook into it.

    Keeps forward_loss in fp32 (outside autocast) as in the single-GPU path.
    """
    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent, mask, ids_restore = self.mae.forward_encoder(x, mask_ratio=0.75)
            pred = self.mae.forward_decoder(latent, ids_restore)
        return self.mae.forward_loss(x, pred, mask)  # fp32


def save_checkpoint(path, epoch, model, optimizer, scheduler, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved → {path}  (epoch {epoch + 1}, loss {loss:.4f})")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt["epoch"]
    loss = ckpt.get("loss")
    print(f"Resumed from {path}  (epoch {epoch + 1}, loss {loss:.4f})")
    return epoch, loss


def train(profile: str | None = None, num_epochs: int = 6, target_loss: float | None = None, resume: str | None = None):
    # torchrun sets LOCAL_RANK / WORLD_SIZE; fall back to single-GPU defaults.
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = mae_vit_ultra_light().to(device)

    # Each GPU runs a full 9216-sample batch (same as single-GPU).
    # Total effective batch = 9216 * world_size; lr scales linearly.
    per_gpu_batch = 1024 * 9
    lr = 1.5e-4 * (per_gpu_batch * world_size / 256)  # linear scaling rule
    dataloader = build_dali_loader(
        "./data/yiddish_lines.npy",
        batch_size=per_gpu_batch,
        num_threads=4,
        device_id=rank,
        shard_id=rank,
        num_shards=world_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, fused=True)

    # Cosine decay with 5% linear warmup — matches the MAE paper's schedule.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(dataloader),
        epochs=num_epochs,
        pct_start=0.05,
        anneal_strategy="cos",
    )

    # Load checkpoint before DDP wrapping so weights are already correct when
    # DDP broadcasts rank-0 parameters to other ranks during construction.
    start_epoch = 0
    if resume:
        start_epoch, _ = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1

    step_module = _TrainStep(model)
    if world_size > 1:
        step_module = DDP(step_module, device_ids=[rank])

    @torch.compile(mode="max-autotune")
    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        loss = step_module(batch)
        loss.backward()
        optimizer.step()
        return loss

    model.train()

    with TrainingLogger(device, num_epochs, len(dataloader), profile, silent=not is_main) as logger:
        try:
            for epoch in range(start_epoch, num_epochs):
                logger.begin_epoch(epoch)
                epoch_loss = torch.zeros(1, device=device)

                for step, batch_data in enumerate(dataloader):
                    batch = batch_data[0]["images"]  # (N, 1, H, W) float32 on GPU
                    loss = train_step(batch)
                    scheduler.step()  # CPU-side lr update — outside compiled graph
                    epoch_loss += loss.detach()
                    logger.on_step()

                avg_loss = (epoch_loss / len(dataloader)).item()  # one sync per epoch
                if world_size > 1:
                    # Average the per-shard losses into one global loss for logging/stopping
                    t = torch.tensor(avg_loss, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.AVG)
                    avg_loss = t.item()

                logger.end_epoch(epoch, avg_loss)
                if is_main:
                    save_checkpoint(CHECKPOINT_PATH, epoch, model, optimizer, scheduler, avg_loss)
                if target_loss is not None and avg_loss <= target_loss:
                    break
        except KeyboardInterrupt:
            if is_main:
                print(f"\nInterrupted. Resume with: --resume {CHECKPOINT_PATH}")

        if profile and is_main:
            with logger.profile_step():
                train_step(batch)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="Profile name — captures a trace to runs/<name>_<timestamp>.pt.trace.json")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--target-loss", type=float, default=None, help="Stop early when avg epoch loss reaches this value")
    parser.add_argument("--resume", default=None, metavar="PATH", help=f"Resume from checkpoint (default path: {CHECKPOINT_PATH})")
    args = parser.parse_args()
    train(profile=args.profile, num_epochs=args.epochs, target_loss=args.target_loss, resume=args.resume)
