"""
vit_extractor.py — Small Vision Transformer as drop-in replacement for HierarchicalCNN.

Architecture:
  1. Patch embedding: split (C, 16, 16) into 16 non-overlapping 4×4 patches,
     flatten each (C×4×4) and project to embed_dim
  2. Learnable position embeddings (one per patch)
  3. TransformerEncoder with Pre-LN (more stable than Post-LN)
  4. Mean-pool over patch tokens → Linear(embed_dim → features_dim) → ReLU

With defaults (embed_dim=256, n_layers=2, n_heads=4, ffn_dim=512):
  ~1.25M parameters in the feature extractor — comparable to HierarchicalCNN (1.29M).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallViT(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        patch_size: int = 4,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        in_channels, H, W = observation_space.shape
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"Board size {H}×{W} must be divisible by patch_size={patch_size}"
        )

        n_patches = (H // patch_size) * (W // patch_size)
        patch_dim = in_channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: stabler gradients
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        # (B, C, H, W) → (B, n_patches, patch_dim)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)

        x = self.patch_proj(x) + self.pos_embed   # (B, n_patches, embed_dim)
        x = self.transformer(x)                    # (B, n_patches, embed_dim)
        x = x.mean(dim=1)                          # (B, embed_dim)
        return self.head(x)                        # (B, features_dim)
