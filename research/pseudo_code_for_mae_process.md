From https://medium.com/@jimcanary/masked-autoencoders-a-simple-yet-powerful-approach-to-self-supervised-vision-learning-0ec9dc849dd2

```
# Pseudo-code for the core MAE process
def mae_forward(image):
  # 1. Patchify image
  patches = patchify(image)

  # 2. Randomly mask patches
  visible_patches, masked_patches = random_masking(patches, mask_ratio=0.75)

  # 3. Encode visible patches (partial tokens)
  encoded = encoder(visible_patches)

  # 4. Decode and reconstruct (full tokens)
  reconstruction = decoder(encoded, mask_tokens)

  return reconstruction
```
