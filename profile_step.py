"""Minimal single-step script for NCU profiling.

No DALI, no scheduler, no epoch loop — just warmup + one captured step.
Uses a small batch so NCU can use --replay-mode kernel (no memory backup overflow).

Usage:
    ncu --set basic -o ncu/<name> python profile_step.py
    ncu --set detailed -o ncu/<name> python profile_step.py
"""
import os
import torch

_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir

from mae import mae_vit_ultra_light

BATCH_SIZE = 256  # small enough for --replay-mode kernel (no memory backup overflow)
WARMUP_STEPS = 5  # let cuBLAS/DALI settle before NCU captures

device = torch.device("cuda")
model = mae_vit_ultra_light().to(device).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, fused=True)
batch = torch.randn(BATCH_SIZE, 1, 32, 512, device=device)

def step():
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _, _ = model(batch, mask_ratio=0.75)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss

# Warmup — let cuBLAS pick algorithms, warm up caches
for _ in range(WARMUP_STEPS):
    step()

torch.cuda.synchronize()

# This is the step NCU captures
loss = step()

torch.cuda.synchronize()
print(f"loss: {loss.item():.4f}")
