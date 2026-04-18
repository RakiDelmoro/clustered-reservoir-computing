"""
Spatiotemporal Encoder for CluSTAR.
Dual-stream encoder: spatial stream (current frame) + temporal stream (frame difference).
Both streams use fixed random orthogonal projections — no training required.
"""

import torch
import torch.nn as nn
import numpy as np


class SpatiotemporalEncoder(nn.Module):
    """
    Dual-stream encoder that produces motion-aware embeddings:
      z_spatial  = W_s @ flatten(I_t)              — "WHAT is it?"
      z_temporal = W_t @ flatten(I_t - I_{t-1})    — "IS IT MOVING?"
      z = [z_spatial ; z_temporal]

    For the first frame (no previous frame), the temporal stream outputs zeros.
    Both W_s and W_t are fixed random orthogonal projections (Johnson-Lindenstrauss).
    """

    def __init__(
        self,
        canvas_size: int = 64,
        spatial_dim: int = 256,
        temporal_dim: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.output_dim = spatial_dim + temporal_dim

        input_dim = canvas_size * canvas_size
        rng = np.random.RandomState(seed)

        w_spatial = rng.randn(spatial_dim, input_dim) / np.sqrt(input_dim)
        self.register_buffer("w_spatial", torch.tensor(w_spatial, dtype=torch.float32))

        rng2 = np.random.RandomState(seed + 1000)
        w_temporal = rng2.randn(temporal_dim, input_dim) / np.sqrt(input_dim)
        self.register_buffer(
            "w_temporal", torch.tensor(w_temporal, dtype=torch.float32)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W]

        Returns:
            encoded: [B, T, output_dim]
        """
        if frames.dim() != 5:
            raise ValueError(
                f"SpatiotemporalEncoder expects [B, T, C, H, W], got shape {frames.shape}"
            )

        B, T, C, H, W = frames.shape
        pixel_dim = H * W * C

        flat = frames.reshape(B, T, pixel_dim)

        z_spatial = torch.matmul(flat, self.w_spatial.T)

        temporal_input = torch.zeros(
            B, T, pixel_dim, device=frames.device, dtype=frames.dtype
        )
        temporal_input[:, 1:, :] = flat[:, 1:, :] - flat[:, :-1, :]

        z_temporal = torch.matmul(temporal_input, self.w_temporal.T)

        encoded = torch.cat([z_spatial, z_temporal], dim=-1)

        return encoded
