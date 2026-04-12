# Training Script Improvements

Improvements grouped by priority.

---

## 1. No LR Scheduler — biggest training quality issue

The MAE paper uses cosine decay with linear warmup. Without it, training runs with constant lr=1.5e-4 for all epochs, missing warmup stabilization and convergence gains from decay.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1.5e-4,
    steps_per_epoch=len(dataloader),
    epochs=num_epochs,
    pct_start=0.05,  # 5% warmup
    anneal_strategy="cos",
)
# in the loop, after optimizer.step():
scheduler.step()
```

---

## 2. No Gradient Clipping

ViT-based models are prone to gradient spikes early in training. The MAE paper clips at `max_norm=1.0`:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 3. No Checkpoint Saving

If training crashes at epoch 5 of 6, everything is lost. Save per epoch:

```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}, f"checkpoints/epoch_{epoch}.pt")
```

---

## 4. No TensorBoard Loss Logging

`SummaryWriter` is imported but never used. Loss curves are far more useful for comparing runs than the rich end-of-training summary. The `TrainingLogger` should write loss/lr to TensorBoard on each step.

---

## 5. Dead Code

`find_monitor_image`, `load_monitor_image`, `log_reconstruction`, `SummaryWriter`, and `cv2` are all defined/imported but never called in `train()`. Either wire them up or remove them (~40 lines).

---

## 6. `forward_loss` Runs Inside `autocast`

`forward_loss` is called inside `model.forward()` which is inside the `torch.autocast` block. The `patchify`, `mean`, `var`, and MSE computation all run with bf16 inputs, reducing loss precision. Moving loss computation outside autocast ensures fp32:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio)
    pred = model.forward_decoder(latent, ids_restore)
loss = model.forward_loss(imgs, pred, mask)  # fp32
```

Requires calling `forward_encoder`, `forward_decoder`, `forward_loss` separately instead of `model(batch)`.

---

## 7. `bnb.AdamW8bit` + `torch.compile` Interaction

8-bit optimizers use custom CUDA kernels. With `mode="reduce-overhead"`, torch.compile tries to use CUDA graphs — these require fully static memory layouts, which the quantized optimizer state may not guarantee. If warnings or slowdowns appear, try `mode="default"` instead.

---

## Priority Order

1 → 2 → 3 → 5 → 4 → 6
