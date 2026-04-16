"""
Clustered Echo State Network Reservoir
Implements: small-world topology + multi-timescale dynamics + input routing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class ClusteredReservoir(nn.Module):
    """
    Structured reservoir with clustered connectivity and multi-timescale neurons.
    Fixed random weights — only state dynamics, no training.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 1000,
        num_clusters: int = 10,
        cluster_connectivity: float = 0.3,
        inter_cluster_connectivity: float = 0.02,
        spectral_radius_global: float = 0.95,
        input_scaling: float = 0.5,
        bias_scaling: float = 0.1,
        activation: str = "tanh",
        seed: int = 42,
        timescale_config: Optional[List[dict]] = None,
        routing_config: Optional[dict] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.num_clusters = num_clusters

        rng = np.random.RandomState(seed)

        # ==================== 1. Neuron Assignment to Clusters ====================
        neurons_per_cluster = reservoir_size // num_clusters
        cluster_assignments = []
        for c in range(num_clusters):
            cluster_assignments.extend([c] * neurons_per_cluster)
        # Handle remainder
        if len(cluster_assignments) < reservoir_size:
            cluster_assignments.extend(
                [0] * (reservoir_size - len(cluster_assignments))
            )
        self.cluster_assignments = torch.tensor(
            cluster_assignments, dtype=torch.long
        )  # [N]

        # ==================== 2. Multi-Timescale Leaking Rates ====================
        if timescale_config is None:
            timescale_config = [
                {
                    "name": "fast",
                    "fraction": 0.3,
                    "alpha_range": [0.2, 0.4],
                    "spectral_radius_range": [0.7, 0.85],
                },
                {
                    "name": "medium",
                    "fraction": 0.4,
                    "alpha_range": [0.5, 0.7],
                    "spectral_radius_range": [0.8, 0.95],
                },
                {
                    "name": "slow",
                    "fraction": 0.3,
                    "alpha_range": [0.8, 1.0],
                    "spectral_radius_range": [0.9, 0.99],
                },
            ]

        self.alpha = torch.zeros(reservoir_size)  # Leaking rates
        self.cluster_spectral_radii = {}  # Per-cluster spectral radius

        neurons_assigned = 0
        for ts_cfg in timescale_config:
            num_neurons = int(reservoir_size * ts_cfg["fraction"])
            idx_start = neurons_assigned
            idx_end = neurons_assigned + num_neurons

            # Sample alpha from range for these neurons
            alphas = rng.uniform(*ts_cfg["alpha_range"], size=num_neurons)
            self.alpha[idx_start:idx_end] = torch.tensor(alphas, dtype=torch.float32)

            # Assign spectral radius to clusters (same for all neurons in cluster)
            # We'll use this later when scaling W_res
            rho_val = rng.uniform(*ts_cfg["spectral_radius_range"])
            for c in range(num_clusters):
                # Spread timescales across all clusters (each cluster has mix)
                if c % 3 == ["fast", "medium", "slow"].index(ts_cfg["name"]):
                    self.cluster_spectral_radii[c] = rho_val
                    break

            neurons_assigned += num_neurons

        # Handle any remaining neurons
        if neurons_assigned < reservoir_size:
            remainder = reservoir_size - neurons_assigned
            alphas = rng.uniform(0.5, 0.7, size=remainder)
            self.alpha[neurons_assigned:] = torch.tensor(alphas, dtype=torch.float32)

        # ==================== 3. Input Weight Matrix (W_in) ====================
        # Sparse input connections: each neuron connects to only some input features
        input_sparsity = 0.3  # Each neuron gets 30% of input dimensions
        W_in = np.zeros((reservoir_size, input_dim))
        for i in range(reservoir_size):
            # Randomly select input features to connect to
            num_connected = int(input_dim * input_sparsity)
            connected_idxs = rng.choice(input_dim, size=num_connected, replace=False)
            # Random weights with scaling
            weights = rng.randn(num_connected) * input_scaling
            W_in[i, connected_idxs] = weights

        self.register_buffer("W_in", torch.tensor(W_in, dtype=torch.float32))

        # ==================== 4. Recurrent Weight Matrix (W_res) - Structured ====================
        # Build using small-world (Watts-Strogatz) per cluster
        W_res = np.zeros((reservoir_size, reservoir_size))

        for cluster_id in range(num_clusters):
            # Get neuron indices in this cluster
            cluster_neurons = (
                (self.cluster_assignments == cluster_id)
                .nonzero(as_tuple=True)[0]
                .numpy()
            )
            n_in_cluster = len(cluster_neurons)

            # Within-cluster connectivity (dense)
            for i in range(n_in_cluster):
                for j in range(n_in_cluster):
                    if i == j:  # No self-connections
                        continue
                    if rng.rand() < cluster_connectivity:
                        # Weight: Gaussian, scaled to desired spectral radius for this cluster
                        rho_target = self.cluster_spectral_radii.get(
                            cluster_id, spectral_radius_global
                        )
                        # Scale by cluster spectral radius
                        scale = rho_target / np.sqrt(
                            n_in_cluster
                        )  # Rough scaling for spectral radius
                        W_res[cluster_neurons[i], cluster_neurons[j]] = (
                            rng.randn() * scale
                        )

            # Inter-cluster sparse long-range connections
            for i in cluster_neurons:
                for j in range(reservoir_size):
                    if self.cluster_assignments[j] == cluster_id:
                        continue  # Skip intra-cluster (already handled)
                    if rng.rand() < inter_cluster_connectivity:
                        # Sparse, small weight
                        W_res[i, j] = rng.randn() * 0.1  # Small scale

        # Enforce spectral radius global constraint
        # Scale entire W_res to have max eigenvalue = spectral_radius_global
        eigenvalues = np.linalg.eigvals(W_res)
        current_rho = max(abs(eigenvalues))
        if current_rho > 0:
            W_res = W_res * (spectral_radius_global / current_rho)

        self.register_buffer("W_res", torch.tensor(W_res, dtype=torch.float32))

        # ==================== 5. Bias ====================
        bias = rng.randn(reservoir_size) * bias_scaling
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

        # ==================== 6. Activation ====================
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ==================== 7. Input Routing Config ====================
        self.routing_enabled = routing_config is not None and routing_config.get(
            "enabled", False
        )
        if self.routing_enabled:
            self.stream_names = routing_config.get(
                "streams", ["position", "velocity", "rotation", "shape"]
            )
            # Map input features to clusters (simplified: use input feature indices)
            # This will be handled at the forward pass level (external to reservoir)
            self.input_stream_info = routing_config.get("stream_dims", {})
            # For now: stream_dims = {"position": 2, "velocity": 2, "rotation": 1, "shape": 7}
        else:
            self.stream_names = []

    @property
    def device(self):
        return self.W_in.device

    def forward(
        self, u: torch.Tensor, x_prev: torch.Tensor, return_cluster_states: bool = False
    ) -> torch.Tensor:
        """
        One-step reservoir update.

        Args:
            u: [batch, input_dim] — input at current timestep
            x_prev: [batch, reservoir_size] — previous state
            return_cluster_states: if True, also return per-cluster activations

        Returns:
            x_new: [batch, reservoir_size] — new state
        """
        # Input + recurrent + bias
        u_part = torch.matmul(u, self.W_in.T)  # [batch, N]
        x_part = torch.matmul(x_prev, self.W_res.T)  # [batch, N]

        # Apply neuron-specific leaking rates (continuous-time integration)
        # x(t+1) = (1-α) * x(t) + α * tanh(u + W_res·x(t) + b)
        pre_activation = u_part + x_part + self.bias.unsqueeze(0)
        h = self.activation(pre_activation)

        # Leaky integration
        alpha = self.alpha.unsqueeze(0)  # [1, N]
        x_new = (1.0 - alpha) * x_prev + alpha * h

        if return_cluster_states:
            # Split by cluster
            cluster_states = {}
            for c in range(self.num_clusters):
                mask = self.cluster_assignments == c
                cluster_states[f"cluster_{c}"] = x_new[:, mask]
            return x_new, cluster_states

        return x_new

    def initialize_state(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize reservoir state (zeros)."""
        return torch.zeros(batch_size, self.reservoir_size, device=self.device)

    def get_cluster_mask(self, cluster_id: int) -> torch.Tensor:
        """Get boolean mask for neurons in a specific cluster."""
        return self.cluster_assignments == cluster_id

    def forward_sequence(
        self, inputs: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run reservoir over a full sequence.

        Args:
            inputs: [batch, time, input_dim]
            initial_state: [batch, reservoir_size] or None

        Returns:
            all_states: [batch, time, reservoir_size]
            final_state: [batch, reservoir_size]
        """
        batch_size, seq_len, _ = inputs.shape
        if initial_state is None:
            x = self.initialize_state(batch_size)
        else:
            x = initial_state

        all_states = []
        for t in range(seq_len):
            x = self.forward(inputs[:, t, :], x)
            all_states.append(x)

        all_states = torch.stack(all_states, dim=1)  # [B, T, N]
        return all_states, x


def test_reservoir():
    """Quick test of CluSTAR reservoir."""
    reservoir = ClusteredReservoir(
        input_dim=128,
        reservoir_size=500,
        num_clusters=5,
        cluster_connectivity=0.3,
        inter_cluster_connectivity=0.02,
        spectral_radius_global=0.95,
        seed=42,
    )

    print(f"Reservoir size: {reservoir.reservoir_size}")
    print(f"Num clusters: {reservoir.num_clusters}")
    print(f"Cluster assignments: {reservoir.cluster_assignments[:10]}")

    # Test forward
    batch_size = 32
    u = torch.randn(batch_size, 128)
    x_prev = reservoir.initialize_state(batch_size)
    x_new = reservoir.forward(u, x_prev)
    print(f"Output state shape: {x_new.shape}")
    assert x_new.shape == (batch_size, 500)

    # Test sequence
    seq = torch.randn(batch_size, 30, 128)
    all_states, final = reservoir.forward_sequence(seq)
    print(f"Sequence states: {all_states.shape}, final: {final.shape}")
    assert all_states.shape == (batch_size, 30, 500)

    print("Reservoir tests passed!")


if __name__ == "__main__":
    test_reservoir()
