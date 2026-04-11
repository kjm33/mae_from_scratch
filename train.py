import os
import time

import cv2
import torch
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn

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
    console = Console()

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
            acc_events=True,
        )
        prof.start()

    num_epochs = 10
    train_start = time.time()
    max_vram_mb = 0.0
    total_loss = 0.0
    total_steps = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[num_epochs]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("loss [green]{task.fields[loss]:.4f}"),
        TextColumn("VRAM [yellow]{task.fields[vram]:.1f} GB"),
        TimeElapsedColumn(),
        console=console,
        disable=not accelerator.is_local_main_process,
    ) as progress:
        task = progress.add_task(
            "train", total=len(dataloader),
            epoch=1, num_epochs=num_epochs, loss=0.0, vram=0.0,
        )

        for epoch in range(num_epochs):
            progress.reset(task, total=len(dataloader))
            progress.update(task, epoch=epoch + 1)
            epoch_loss = 0.0
            epoch_start = time.time()

            for step, batch in enumerate(dataloader):
                batch = batch.to(accelerator.device, non_blocking=True)
                batch = batch.to(torch.bfloat16).div_(255.0)

                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = model(batch, mask_ratio=0.75) # high CPU usage - due to torch.compilation

                accelerator.backward(loss) # CPU peak
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                total_loss += loss_val
                total_steps += 1

                if accelerator.is_local_main_process:
                    vram_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1024**2
                    max_vram_mb = max(max_vram_mb, vram_mb)
                    progress.update(task, advance=1, loss=loss_val, vram=vram_mb / 1024)

                if prof is not None:
                    prof.step()
                    if step >= 4:  # wait(1) + warmup(1) + active(3) = 5 steps
                        prof.stop()
                        prof = None

            if accelerator.is_local_main_process:
                epoch_time = time.time() - epoch_start
                avg_loss = epoch_loss / (step + 1)
                console.print(
                    f"[bold]Epoch {epoch + 1}/{num_epochs}[/bold] "
                    f"avg_loss=[green]{avg_loss:.4f}[/green] "
                    f"time=[cyan]{epoch_time:.1f}s[/cyan]"
                )

    if accelerator.is_local_main_process:
        elapsed = time.time() - train_start
        table = Table(title=f"Training Summary — {TENSORBOARD_PROFILE}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_row("Total time", f"{elapsed / 3600:.2f} h  ({elapsed:.0f} s)")
        table.add_row("Total steps", str(total_steps))
        table.add_row("Avg steps/sec", f"{total_steps / elapsed:.2f}")
        table.add_row("Peak VRAM", f"{max_vram_mb / 1024:.2f} GB  ({max_vram_mb:.0f} MB)")
        table.add_row("Avg loss", f"{total_loss / total_steps:.4f}")
        console.print(table)


if __name__ == "__main__":
    train()
