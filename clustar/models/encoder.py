"""
Spatial Encoder for CluSTAR.
Random projection layer that preserves spatial correlations.
"""

import torch
import torch.nn as nn
import numpy as np


class RandomOrthogonalProjection(nn.Module):
    """
    Fixed random orthogonal projection for dimensionality reduction.
    Preserves inner-product structure (like Johnson-Lindenstrauss).
    """

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)

        # Random projection matrix (Johnson-Lindenstrauss)
        # Weight shape: [output_dim, input_dim] for multiplication: x @ W.T
        weight = rng.randn(output_dim, input_dim) / np.sqrt(input_dim)

        # Fixed weights — not trainable
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim] or [..., H, W] (will flatten)

        Returns:
            [..., output_dim]
        """
        if x.dim() == 3:  # [B, H, W]
            x = x.flatten(start_dim=1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return torch.matmul(x, self.weight.T)


class SpatialEncoder(nn.Module):
    """
    Encapsulates spatial encoding: flatten + random projection.
    """

    def __init__(
        self,
        img_size: int = 28,
        canvas_size: int = 64,
        output_dim: int = 128,
        encoder_type: str = "random_orthogonal",
        seed: int = 42,
    ):
        super().__init__()
        self.img_size = img_size
        self.canvas_size = canvas_size
        self.output_dim = output_dim

        if encoder_type == "random_orthogonal":
            # Input is flattened frame (canvas_size^2)
            input_dim = canvas_size * canvas_size
            self.encoder = RandomOrthogonalProjection(
                input_dim=input_dim, output_dim=output_dim, seed=seed
            )
        elif encoder_type == "random_normal":
            # Simple random weights (Gaussian)
            input_dim = canvas_size * canvas_size
            weight = torch.randn(output_dim, input_dim) * (1.0 / np.sqrt(input_dim))
            self.register_buffer("weight", weight)
            self.encoder = lambda x: torch.matmul(x.flatten(start_dim=1), self.weight.T)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W] or [B, C, H, W] or [C, H, W]

        Returns:
            encoded: [B, T, output_dim] or [B, output_dim] or [output_dim]
        """
        original_dim = frames.dim()
        if original_dim == 5:  # [B, T, C, H, W]
            B, T = frames.shape[0], frames.shape[1]
            frames = frames.flatten(start_dim=0, end_dim=1)  # [B*T, C, H, W]

        # Flatten spatial dimensions
        batch_size = frames.shape[0]
        flat_frames = frames.view(batch_size, -1)  # [B_flat, H*W]

        encoded = self.encoder(flat_frames)  # [B_flat, output_dim]

        if original_dim == 5:
            encoded = encoded.view(B, T, -1)  # [B, T, output_dim]
        elif original_dim == 4:
            pass  # [B, output_dim]
        elif original_dim == 3:
            pass  # [output_dim]

        return encoded


def test_encoder():
    """Quick test."""
    encoder = SpatialEncoder(
        img_size=28, canvas_size=64, output_dim=128, encoder_type="random_orthogonal"
    )
    # Test: batch of sequences
    x = torch.randn(32, 30, 1, 64, 64)
    y = encoder(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (32, 30, 128)

    # Test: single frame batch
    x2 = torch.randn(32, 1, 64, 64)
    y2 = encoder(x2)
    print(f"Single frame batch: {x2.shape} -> {y2.shape}")
    assert y2.shape == (32, 128)

    print("Encoder tests passed!")


if __name__ == "__main__":
    test_encoder()
