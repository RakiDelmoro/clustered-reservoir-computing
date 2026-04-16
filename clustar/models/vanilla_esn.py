"""
Vanilla Echo State Network Baseline.
Standard unstructured ESN for comparison against CluSTAR.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class VanillaESN(nn.Module):
    """
    Standard Echo State Network with random Erdős–Rényi connectivity.
    Baseline for comparison.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 1000,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        bias_scaling: float = 0.1,
        sparsity: float = 0.1,  # 10% connectivity
        activation: str = "tanh",
        leaking_rate: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()

        rng = np.random.RandomState(seed)
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate

        # Input weights: sparse random
        W_in = rng.randn(reservoir_size, input_dim) * input_scaling
        self.register_buffer("W_in", torch.tensor(W_in, dtype=torch.float32))

        # Reservoir weights: random sparse matrix
        # Initialize with sparse random values
        W_res = np.zeros((reservoir_size, reservoir_size))
        for i in range(reservoir_size):
            for j in range(reservoir_size):
                if i != j and rng.rand() < sparsity:
                    W_res[i, j] = rng.randn()

        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        current_rho = max(abs(eigenvalues))
        if current_rho > 0:
            W_res = W_res * (spectral_radius / current_rho)

        self.register_buffer("W_res", torch.tensor(W_res, dtype=torch.float32))

        # Bias
        bias = rng.randn(reservoir_size) * bias_scaling
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

        # Activation
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, u: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        One-step update: x(t+1) = (1-α)x(t) + α * f(W_in u(t) + W_res x(t) + b)
        """
        u_part = torch.matmul(u, self.W_in.T)
        x_part = torch.matmul(x_prev, self.W_res.T)
        pre_activation = u_part + x_part + self.bias.unsqueeze(0)
        h = self.activation(pre_activation)

        # Leaky integration
        x_new = (1.0 - self.leaking_rate) * x_prev + self.leaking_rate * h
        return x_new

    def forward_sequence(
        self, inputs: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run ESN over sequence.

        Args:
            inputs: [B, T, input_dim]

        Returns:
            all_states: [B, T, reservoir_size]
            final_state: [B, reservoir_size]
        """
        B, T, _ = inputs.shape
        if initial_state is None:
            x = torch.zeros(B, self.reservoir_size, device=inputs.device)
        else:
            x = initial_state

        all_states = []
        for t in range(T):
            x = self.forward(inputs[:, t, :], x)
            all_states.append(x)

        all_states = torch.stack(all_states, dim=1)
        return all_states, x


def test_vanilla_esn():
    """Test vanilla ESN."""
    esn = VanillaESN(input_dim=128, reservoir_size=500, seed=42)

    batch = torch.randn(32, 30, 128)
    states, final = esn.forward_sequence(batch)
    print(f"States shape: {states.shape}, final: {final.shape}")
    assert states.shape == (32, 30, 500)

    print("Vanilla ESN test passed!")


if __name__ == "__main__":
    test_vanilla_esn()
