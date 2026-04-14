# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
#
# Masked Autoencoder (MAE) for self-supervised vision learning.
# The model masks a large fraction of image patches, encodes only the visible
# patches with a ViT encoder, then reconstructs the full image with a lightweight
# decoder. Reconstruction loss is computed only on masked patches.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed


class FlashAttention(nn.Module):
    """Drop-in replacement for timm's Attention that uses PyTorch's
    scaled_dot_product_attention (Flash Attention 2 when available)."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, attn_mask=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # PyTorch F.scaled_dot_product_attention automatically uses Flash Attention 2
        # when the mask is compatible or None.
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=False,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer (ViT) backbone.

    High-level flow:
      1. Split image into patches and embed them.
      2. Randomly mask most patches (e.g. 75%); keep only a subset visible.
      3. Encode visible patches with a ViT encoder (no masked tokens in encoder).
      4. Decoder gets full sequence: re-insert mask tokens, unshuffle to original order.
      5. Decoder predicts pixel values per patch; loss is MSE on masked patches only.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --- Input / grid setup ---
        self.in_chans = in_chans
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)
        ph, pw = self.patch_size
        if isinstance(img_size, (list, tuple)):
            self._grid_size = (img_size[0] // ph, img_size[1] // pw)
        else:
            self._grid_size = (img_size // ph, img_size // pw)

        # --------------------------------------------------------------------------
        # MAE encoder: processes only the *visible* (non-masked) patches
        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attn_layer=FlashAttention)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder: reconstructs full image from encoder output + mask tokens
        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attn_layer=FlashAttention)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, ph * pw * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize encoder/decoder weights. Position embeddings use fixed 2D sin-cos; rest use standard init."""
        grid_size = self._grid_size

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Flatten image into a sequence of patch vectors.

        Args:
            imgs: (N, C, H, W)
        Returns:
            x: (N, L, patch_size**2 * C)
        """
        ph, pw = self.patch_size
        c = self.in_chans
        assert imgs.shape[2] % ph == 0 and imgs.shape[3] % pw == 0

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw
        x = imgs.reshape(shape=(imgs.shape[0], c, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph * pw * c))
        return x

    def unpatchify(self, x):
        """
        Convert patch sequence back to image shape. Inverse of patchify.

        Args:
            x: (N, L, patch_size**2 * C)
        Returns:
            imgs: (N, C, H, W)
        """
        ph, pw = self.patch_size
        c = self.in_chans
        h, w = self._grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Randomly mask a fraction of patches per sample.

        Args:
            x: (N, L, D) -- patch sequence (no cls token yet)
            mask_ratio: fraction of patches to mask (e.g. 0.75 -> keep 25%)
        Returns:
            x_masked: (N, len_keep, D)
            mask: (N, L) -- 0 = kept, 1 = masked
            ids_restore: (N, L) -- indices to unshuffle back to full order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Encode only the visible (non-masked) patches."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decode full sequence and predict patch pixels."""
        x = self.decoder_embed(x)

        num_mask = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        MSE reconstruction loss on masked patches only.
        Optionally normalizes patch pixels (norm_pix_loss) for more stable training.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (N, L) -- MSE per patch

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Full MAE forward: encode (with masking), decode, compute reconstruction loss.

        Returns:
            loss: scalar reconstruction loss (on masked patches)
            pred: (N, L, p*p*C)
            mask: (N, L) -- binary mask (1 = masked)
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def mae_vit_base_patch16(**kwargs):
    """MAE with ViT-Base: 768-dim, 12 layers, 12 heads; decoder 512-dim, 8 blocks."""
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_large_patch16(**kwargs):
    """MAE with ViT-Large: 1024-dim, 24 layers, 16 heads; decoder 512-dim, 8 blocks."""
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_huge_patch14(**kwargs):
    """MAE with ViT-Huge: 1280-dim, 32 layers, 16 heads, 14x14 patches; decoder 512-dim, 8 blocks."""
    return MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_base_patch32x8_32x512(**kwargs):
    """MAE for Yiddish text/grayscale: img 32x512, patch (32h,8w), 1 channel, ViT-Base."""
    return MaskedAutoencoderViT(
        img_size=(32, 512), patch_size=(32, 8), in_chans=1,
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=True, **kwargs)

def mae_vit_ultra_light(**kwargs):
    """MAE for Yiddish text/grayscale: img 32x512, patch (32h,8w), 1 channel, ViT-Tiny?."""
    return MaskedAutoencoderViT(
        img_size=(32, 512),
        patch_size=(32, 8),
        in_chans=1,
        embed_dim=256,
        depth=6,
        num_heads=4,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=True, **kwargs)