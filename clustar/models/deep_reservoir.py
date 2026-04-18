"""
Deep Reservoir: Two-layer clustered reservoir for spatiotemporal processing.
Layer 1 captures spatial features; Layer 2 captures temporal dynamics from state changes.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from clustar.models.reservoir import ClusteredReservoir


class DeepReservoir(nn.Module):
    """
    Two-layer reservoir architecture:
      Layer 1: processes encoded spatial features (standard reservoir)
      Layer 2: processes [state_t ; delta_t] to capture temporal dynamics

    All weights are fixed random — only the readout is trained.
    """

    def __init__(
        self,
        layer1: ClusteredReservoir,
        layer2: ClusteredReservoir,
    ):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

        self.reservoir_size_L1 = layer1.reservoir_size
        self.reservoir_size_L2 = layer2.reservoir_size
        self.num_clusters_L1 = layer1.num_clusters
        self.num_clusters_L2 = layer2.num_clusters

    def forward(
        self,
        encoded: torch.Tensor,
        state_L1: torch.Tensor,
        state_L2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward for autoregressive inference.

        Args:
            encoded: [batch, input_dim] — spatial encoder output
            state_L1: [batch, reservoir_size_L1]
            state_L2: [batch, reservoir_size_L2]

        Returns:
            new_state_L1: [batch, reservoir_size_L1]
            new_state_L2: [batch, reservoir_size_L2]
        """
        new_state_L1 = self.layer1(encoded, state_L1)
        delta = new_state_L1 - state_L1
        input_L2 = torch.cat([new_state_L1, delta], dim=1)
        new_state_L2 = self.layer2(input_L2, state_L2)
        return new_state_L1, new_state_L2

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        initial_state_L1: Optional[torch.Tensor] = None,
        initial_state_L2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run both layers over a full sequence.

        Args:
            inputs: [batch, time, input_dim] — encoded spatial features
            initial_state_L1: [batch, reservoir_size_L1] or None
            initial_state_L2: [batch, reservoir_size_L2] or None

        Returns:
            states_L1: [batch, time, reservoir_size_L1]
            states_L2: [batch, time, reservoir_size_L2]
            final_state_L1: [batch, reservoir_size_L1]
            final_state_L2: [batch, reservoir_size_L2]
        """
        states_L1, final_L1 = self.layer1.forward_sequence(inputs, initial_state_L1)

        delta = torch.zeros_like(states_L1)
        delta[:, 0, :] = states_L1[:, 0, :]
        delta[:, 1:, :] = states_L1[:, 1:, :] - states_L1[:, :-1, :]

        input_L2 = torch.cat([states_L1, delta], dim=-1)

        states_L2, final_L2 = self.layer2.forward_sequence(input_L2, initial_state_L2)

        return states_L1, states_L2, final_L1, final_L2

    def initialize_states(
        self, batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize both layer states to zeros."""
        state_L1 = self.layer1.initialize_state(batch_size)
        state_L2 = self.layer2.initialize_state(batch_size)
        return state_L1, state_L2

    @property
    def device(self):
        return self.layer1.device
