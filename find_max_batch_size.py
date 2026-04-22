"""Binary-search for the largest batch size that fits in GPU memory.

Tests the full training step (forward + backward + optimizer step) with bf16
autocast, matching train.py exactly. No torch.compile — we want fast probing,
not kernel benchmarking.

Usage:
    python find_max_batch_size.py
    python find_max_batch_size.py --model mae_vit_ultra_light
    python find_max_batch_size.py --device cuda:1 --min 64 --max 8192
"""

import argparse
import gc

import torch

import mae as mae_module


MODEL_FACTORIES = {
    "mae_vit_ultra_light": mae_module.mae_vit_ultra_light,
    "mae_vit_small_patch32x8": mae_module.mae_vit_small_patch32x8,
}

IMG_SIZE = (32, 512)
IN_CHANS = 1
MASK_RATIO = 0.75


def try_batch_size(model, optimizer, batch_size: int, device: torch.device) -> bool:
    """Return True if a full train step fits; False on OOM."""
    x = torch.randn(batch_size, IN_CHANS, *IMG_SIZE, device=device)
    try:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=MASK_RATIO)
            pred = model.forward_decoder(latent, ids_restore)
        loss = model.forward_loss(x, pred, mask)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del x
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch_size(model_name: str, device: torch.device, lo: int, hi: int) -> int:
    model = MODEL_FACTORIES[model_name]().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}  ({param_count:,} parameters)")
    print(f"Device: {device}  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"Search range: [{lo}, {hi}]")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, fused=True)
    model.train()

    # Warm-up at lo to initialize optimizer state so memory estimates are accurate.
    print(f"Warm-up at batch_size={lo} ...", end=" ", flush=True)
    if not try_batch_size(model, optimizer, lo, device):
        print("FAIL — even minimum batch size OOMs. Lower --min.")
        return -1
    print("OK")
    print()

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        vram_used = torch.cuda.memory_allocated(device) / 1e9
        print(f"  Trying {mid:>6d} ... (VRAM before: {vram_used:.2f} GB)", end=" ", flush=True)
        if try_batch_size(model, optimizer, mid, device):
            best = mid
            print(f"OK   (peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB)")
            lo = mid + 1
        else:
            print("OOM")
            hi = mid - 1
        torch.cuda.reset_peak_memory_stats(device)

    return best


def main():
    parser = argparse.ArgumentParser(description="Find max training batch size via binary search.")
    parser.add_argument("--model", default="mae_vit_small_patch32x8", choices=list(MODEL_FACTORIES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min", type=int, default=64, dest="min_bs", metavar="N")
    parser.add_argument("--max", type=int, default=16384, dest="max_bs", metavar="N")
    args = parser.parse_args()

    device = torch.device(args.device)
    result = find_max_batch_size(args.model, device, args.min_bs, args.max_bs)
    if result > 0:
        print(f"\nMax batch size: {result}")


if __name__ == "__main__":
    main()
