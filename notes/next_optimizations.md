# Next Optimizations

GPU kernel density is already 99.7% and two CUDA graphs fire per step. The GPU is not the
bottleneck anymore. Further throughput gains require more work-per-step (bigger batch) or
more GPUs. The bigger gap right now is training quality — no LR schedule, no gradient
clipping, no checkpointing, no TensorBoard monitoring.

---

## 1. Increase batch size: 256 → 512  *(~1.5–1.7× more samples/sec)*

7.4 GB VRAM headroom (16.56 / 24 GB used). Doubling batch trades slightly longer steps for
roughly 1.5–1.7× more samples/sec (GPU already saturated, so matmul utilization improves).

**Linear LR scaling rule:** new lr = `1.5e-4 × (512/256) = 3.0e-4`.

```python
# train.py
batch_size = 512
lr = 3.0e-4
dataloader = build_dali_loader("./data/yiddish_lines.npy", batch_size=batch_size, ...)
```

If OOM at 512, try 384 first (`lr = 2.25e-4`).

---

## 2. LR schedule: linear warmup + cosine decay  *(critical for long runs)*

MAE paper trains 1600 epochs with warmup + cosine decay. Constant lr past ~10 epochs
is suboptimal.

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=lr,
    total_steps=num_epochs * len(dataloader),
    pct_start=0.05,       # 5% warmup
    anneal_strategy="cos",
    div_factor=25,        # start lr = max_lr / 25
    final_div_factor=1e4,
)

# in training loop, after train_step():
scheduler.step()
```

Call `scheduler.step()` outside the compiled function — no effect on graph capture.

---

## 3. Gradient clipping (max_norm=1.0)  *(prevents early ViT loss spikes)*

Standard for ViT pretraining. Add inside `train_step`, between backward and optimizer.step():

```python
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
```

Note: `clip_grad_norm_` forces an additional graph break, splitting into three compiled
graphs instead of two. Overhead ~0.3 ms/step — acceptable.

---

## 4. Wire up TensorBoard logging  *(SummaryWriter already imported but unused)*

```python
writer = SummaryWriter(LOG_DIR)

# after end_epoch:
writer.add_scalar("train/loss", avg_loss, epoch)
writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

monitor_img = load_monitor_image(find_monitor_image("./data/yiddish_lines"), IMG_SIZE, device)
if monitor_img is not None:
    log_reconstruction(writer, model, monitor_img, epoch)

writer.close()
```

`log_reconstruction()` is already implemented in train.py — just never called.

---

## 5. Move loss computation outside autocast  *(fp32 precision for norm_pix_loss)*

`forward_loss()` (patchify → mean/var → MSE) currently runs in bf16 because it's inside
the autocast context. `norm_pix_loss=True` does variance normalization — imprecise in bf16.

`mae/model.py` already has `forward_encoder`, `forward_decoder`, `forward_loss` as separate
methods. Restructure `train_step`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    latent, mask, ids_restore = model.forward_encoder(batch, mask_ratio=0.75)
    pred = model.forward_decoder(latent, ids_restore)
loss = model.forward_loss(batch, pred, mask)  # runs in fp32
```

---

## 6. Checkpoint saving  *(no crash recovery currently)*

```python
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 10  # epochs

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# after end_epoch:
if (epoch + 1) % SAVE_EVERY == 0:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": avg_loss,
    }, f"{CHECKPOINT_DIR}/epoch_{epoch+1:04d}.pt")
```

---

## Out of Scope (larger refactor)

**Multi-GPU DDP (2× RTX 3090 available):** Would ~2× throughput. Requires:
- `torchrun` launcher + `dist.init_process_group`
- `DistributedDataParallel` wrapper
- Two DALI pipeline instances, each processing half of `epoch_size`
- LR scaling: effective batch = 512 × 2

Profile single-GPU with items 1–5 first, then consider DDP.

---

## Verification

After implementing:
1. `python train.py` — no OOM, loss decreases smoothly
2. `tensorboard --logdir ./runs/mae_yiddish` — loss + lr curves visible
3. `checkpoints/` directory populated after epoch 10
4. Set `TENSORBOARD_PROFILE = "8_batch512_schedule"` and run
   `python analyze_trace.py runs/8_batch512_schedule/T-*.pt.trace.json`
   — confirm kernel density stays ≥99%
