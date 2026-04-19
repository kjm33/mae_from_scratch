# Architecture Optimization Opportunities

Analysis of `mae_vit_ultra_light` — embed_dim=256, depth=6, num_heads=4,
decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=8, patch_size=(32,8), 64 tokens.

---

## 1. `forward_loss` runs inside autocast — quality issue

**The problem:**
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss, _, _ = model(batch, mask_ratio=0.75)  # forward_loss() runs here in bf16
```
Inside `forward_loss`, `target.var(dim=-1)` computes variance over 256 pixels per patch in
bf16. bf16 has only 7 mantissa bits — variance of small values is imprecise. For Yiddish
text patches (lots of near-white background with low variance), this error is amplified by
the `/ (var + 1e-6)**.5` normalization, producing unstable loss values.

**Fix:** split the forward and pull loss computation out of autocast:
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    latent, mask, ids_restore = model.forward_encoder(batch, mask_ratio=0.75)
    pred = model.forward_decoder(latent, ids_restore)
loss = model.forward_loss(batch, pred, mask)  # fp32
```

---

## 2. `.repeat` for gather index expansion — large hidden allocations

In `random_masking`:
```python
index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
```
At B=8192, len_keep=16, D=256: allocates a **256 MB** int64 tensor every forward pass.

In `forward_decoder`:
```python
index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
```
At B=8192, L=64, decoder_dim=128: allocates a **512 MB** int64 tensor every forward pass.

Both can be replaced with `.expand` — zero-copy broadcast view. `torch.gather` accepts
non-contiguous tensors, and `torch.compile` fuses expand+gather into a single Triton kernel:

```python
# random_masking
x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

# forward_decoder
x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
```
768 MB of allocations per forward pass reduced to zero.

---

## 3. Decoder `head_dim=16` — below SDPA optimal alignment

`decoder_embed_dim=128, decoder_num_heads=8` → `head_dim=16`.

SDPA and FlashAttention tile QK^T in blocks of 16 or 32. head_dim=16 is a degenerate
single-tile case. Hardware-preferred values: 32, 64, 128.

Fix: `decoder_num_heads=8 → 4` gives `head_dim=32` with no other changes — same total
parameter count, better kernel utilization. The decoder is a reconstruction head, 4 heads
is more than sufficient.

---

## 4. Dead `self.scale` in FlashAttention

```python
self.scale = self.head_dim ** -0.5  # set but never used
```
`F.scaled_dot_product_attention` computes its own scale internally. This attribute is set
but referenced nowhere. Not harmful, just dead code.

---

## 5. CLS token goes through the decoder and is thrown away

```python
# end of forward_decoder:
x = self.decoder_pred(x)
x = x[:, 1:, :]  # cls prediction discarded
```
The CLS token is processed through both decoder blocks (part of the 65×65 attention matrix)
and then its prediction is thrown away. For MAE pretraining on OCR lines with no downstream
classification, it is pure overhead.

Removing it:
- Encoder attention: 17×17 → 16×16 (−11% in QK^T)
- Decoder attention: 65×65 → 64×64 (−3%)

Architectural decision — keep if downstream classification on CLS representation is planned.

---

## 6. `norm_pix_loss` normalization — two memory passes

```python
mean = target.mean(dim=-1, keepdim=True)
var  = target.var(dim=-1, keepdim=True)
target = (target - mean) / (var + 1.e-6)**.5
```
This is exactly what `F.layer_norm` does but less efficiently — mean and var are two separate
memory passes. `F.layer_norm` uses a fused single-pass Welford algorithm:
```python
target = F.layer_norm(target, [target.shape[-1]], eps=1e-6)
```

---

## 7. Decoder `mlp_ratio=4` — oversized FFN

Decoder: depth=2, embed_dim=128, mlp_ratio=4 → hidden=512. The decoder's only job is
mapping 128-dim tokens to 256 pixel values. An FFN that expands 128→512→128 is larger than
the prediction head itself (Linear(128, 256)). `mlp_ratio=2` (hidden=256) would halve
decoder FFN compute with likely no quality loss.

---

## Priority summary

| Issue | Type | Impact | Effort |
|---|---|---|---|
| `forward_loss` inside autocast | Quality (bf16 var precision) | High | Low |
| `.repeat` → `.expand` in gather | Memory (768 MB/step) | High | Low |
| `decoder_num_heads` 8→4 | Kernel efficiency (head_dim 16→32) | Medium | Trivial |
| Dead `self.scale` in FlashAttention | Code cleanliness | Negligible | Trivial |
| CLS token removal | Architecture | Medium | Medium |
| `norm_pix_loss` → `F.layer_norm` | Minor perf + cleanliness | Low | Low |
| Decoder `mlp_ratio` 4→2 | Parameters + compute | Low–Medium | Trivial |
