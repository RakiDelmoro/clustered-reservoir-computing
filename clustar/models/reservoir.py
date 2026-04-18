"""
Clustered Echo State Network Reservoir
Small-world topology + multi-timescale dynamics + mixed activation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class ClusteredReservoir(nn.Module):
    """
    Structured reservoir with clustered connectivity and multi-timescale neurons.
    Mixed activation: fast neurons use ReLU, medium/slow use tanh.
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
        seed: int = 42,
        timescale_config: Optional[List[dict]] = None,
        activation: str = "mixed",
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
        if len(cluster_assignments) < reservoir_size:
            cluster_assignments.extend(
                [0] * (reservoir_size - len(cluster_assignments))
            )
        self.cluster_assignments = torch.tensor(cluster_assignments, dtype=torch.long)

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

        self.register_buffer("alpha", torch.zeros(reservoir_size))
        self.cluster_spectral_radii = {}

        neurons_assigned = 0
        for ts_cfg in timescale_config:
            num_neurons = int(reservoir_size * ts_cfg["fraction"])
            idx_start = neurons_assigned
            idx_end = neurons_assigned + num_neurons

            alphas = rng.uniform(*ts_cfg["alpha_range"], size=num_neurons)
            self.alpha[idx_start:idx_end] = torch.tensor(alphas, dtype=torch.float32)

            rho_val = rng.uniform(*ts_cfg["spectral_radius_range"])
            for c in range(num_clusters):
                if c % 3 == ["fast", "medium", "slow"].index(ts_cfg["name"]):
                    self.cluster_spectral_radii[c] = rho_val
                    break

            neurons_assigned += num_neurons

        if neurons_assigned < reservoir_size:
            remainder = reservoir_size - neurons_assigned
            alphas = rng.uniform(0.5, 0.7, size=remainder)
            self.alpha[neurons_assigned:] = torch.tensor(alphas, dtype=torch.float32)

        # ==================== 3. Activation Function Assignment ====================
        self.activation_mode = activation
        if activation == "tanh":
            relu_mask = torch.zeros(reservoir_size, dtype=torch.bool)
        elif activation == "relu":
            relu_mask = torch.ones(reservoir_size, dtype=torch.bool)
        elif activation == "mixed":
            # Fast neurons (lowest 30% alpha) → ReLU; medium/slow → tanh
            alpha_arr = self.alpha.cpu().numpy()
            threshold = np.percentile(alpha_arr, 30)
            relu_mask = torch.tensor(alpha_arr <= threshold, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.register_buffer("relu_mask", relu_mask)

        # ==================== 4. Input Weight Matrix (W_in) ====================
        input_sparsity = 0.3
        W_in = np.zeros((reservoir_size, input_dim))
        for i in range(reservoir_size):
            num_connected = int(input_dim * input_sparsity)
            connected_idxs = rng.choice(input_dim, size=num_connected, replace=False)
            weights = rng.randn(num_connected) * input_scaling
            W_in[i, connected_idxs] = weights

        self.register_buffer("W_in", torch.tensor(W_in, dtype=torch.float32))

        # ==================== 5. Recurrent Weight Matrix (W_res) ====================
        W_res = np.zeros((reservoir_size, reservoir_size))

        for cluster_id in range(num_clusters):
            cluster_neurons = (
                (self.cluster_assignments == cluster_id)
                .nonzero(as_tuple=True)[0]
                .numpy()
            )
            n_in_cluster = len(cluster_neurons)

            for i in range(n_in_cluster):
                for j in range(n_in_cluster):
                    if i == j:
                        continue
                    if rng.rand() < cluster_connectivity:
                        rho_target = self.cluster_spectral_radii.get(
                            cluster_id, spectral_radius_global
                        )
                        scale = rho_target / np.sqrt(n_in_cluster)
                        W_res[cluster_neurons[i], cluster_neurons[j]] = (
                            rng.randn() * scale
                        )

            for i in cluster_neurons:
                for j in range(reservoir_size):
                    if self.cluster_assignments[j] == cluster_id:
                        continue
                    if rng.rand() < inter_cluster_connectivity:
                        W_res[i, j] = rng.randn() * 0.1

        eigenvalues = np.linalg.eigvals(W_res)
        current_rho = max(abs(eigenvalues))
        if current_rho > 0:
            W_res = W_res * (spectral_radius_global / current_rho)

        self.register_buffer("W_res", torch.tensor(W_res, dtype=torch.float32))

        # ==================== 6. Bias ====================
        bias = rng.randn(reservoir_size) * bias_scaling
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    @property
    def device(self):
        return self.W_in.device

    def forward(self, u: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        One-step reservoir update.

        Args:
            u: [batch, input_dim]
            x_prev: [batch, reservoir_size]

        Returns:
            x_new: [batch, reservoir_size]
        """
        u_part = torch.matmul(u, self.W_in.T)
        x_part = torch.matmul(x_prev, self.W_res.T)

        pre_activation = u_part + x_part + self.bias.unsqueeze(0)

        h = torch.empty_like(pre_activation)
        h[:, self.relu_mask] = torch.relu(pre_activation[:, self.relu_mask])
        h[:, ~self.relu_mask] = torch.tanh(pre_activation[:, ~self.relu_mask])

        alpha = self.alpha.unsqueeze(0)
        x_new = (1.0 - alpha) * x_prev + alpha * h
        x_new = x_new.clamp(-10.0, 10.0)

        return x_new

    def initialize_state(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize reservoir state (zeros)."""
        return torch.zeros(batch_size, self.reservoir_size, device=self.device)

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

        all_states = torch.stack(all_states, dim=1)
        return all_states, x
