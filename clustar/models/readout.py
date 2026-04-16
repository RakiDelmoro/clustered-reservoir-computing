"""
Multi-Task Readout Heads for CluSTAR.
Linear readouts trained via ridge regression (closed-form).
"""

import torch
import math
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiTaskReadout(nn.Module):
    """
    Collection of linear readout heads for different pretext tasks.
    All trained via ridge regression (no gradients during training).
    """

    def __init__(self, reservoir_size: int, task_configs: Dict, device: str = "cpu"):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.task_configs = task_configs
        self.device = device

        # Create a linear layer for each enabled task
        self.heads = nn.ModuleDict()
        for task_name, cfg in task_configs.items():
            if cfg.get("enabled", False):
                if task_name == "frame_prediction":
                    out_dim = 784  # 28x28 flattened
                elif task_name == "segmentation":
                    out_dim = 784  # pixel-wise mask
                elif task_name == "speed_regression":
                    out_dim = 1
                elif task_name == "temporal_order":
                    out_dim = 1  # binary classification
                elif task_name == "rotation_contrast":
                    # Contrastive: embed to space, use cosine similarity
                    out_dim = 64  # projection dimension
                elif task_name == "action_classification":
                    out_dim = cfg.get("num_classes", 4)  # default 4 actions
                else:
                    raise ValueError(f"Unknown task: {task_name}")

                self.heads[task_name] = nn.Linear(reservoir_size, out_dim, bias=True)

        # Initialize weights (will be overwritten by ridge solution)
        for layer in self.heads.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, task: str) -> torch.Tensor:
        """
        Forward pass for a specific task.

        Args:
            state: [batch, reservoir_size] or [batch, seq, reservoir_size]
            task: task name

        Returns:
            output: task-specific prediction
        """
        if task not in self.heads:
            raise ValueError(f"Task {task} not enabled or doesn't exist")

        # Handle temporal dimension
        if state.dim() == 3:
            # Use last timestep state by default
            state = state[:, -1, :]

        return self.heads[task](state)

    def get_weight_matrix(self, task: str) -> torch.Tensor:
        """Get weight matrix for a task (for ridge regression)."""
        return self.heads[task].weight.data

    def set_weight_matrix(self, task: str, W: torch.Tensor, b: torch.Tensor):
        """Set weight matrix after ridge regression."""
        self.heads[task].weight.data = W
        self.heads[task].bias.data = b


class RidgeRegression:
    """
    Closed-form ridge regression solver for linear readout training.
    Solves: min ||Y - XW||^2 + λ||W||^2
    Solution: W = (X^T X + λI)^(-1) X^T Y
    """

    @staticmethod
    def solve(
        X: torch.Tensor, Y: torch.Tensor, lambd: float = 1e-6, bias: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve ridge regression via normal equations.
        Solves: (X_aug^T X_aug + λ_reg) W_aug = X_aug^T Y
        where λ_reg is diagonal with λ for all weights except bias if bias=True.
        """
        device = X.device
        dtype = X.dtype
        lambd = float(lambd)

        # Augment X with bias column
        if bias:
            X_aug = torch.cat(
                [X, torch.ones(X.shape[0], 1, device=device, dtype=dtype)], dim=1
            )
        else:
            X_aug = X

        feat = X_aug.shape[1]

        # Compute X^T X
        XtX = X_aug.t() @ X_aug  # [feat, feat]

        # Build regularization diagonal (no bias)
        reg_diag = torch.full((feat,), lambd, device=device, dtype=dtype)
        if bias:
            reg_diag[-1] = 0.0
        reg = torch.diag(reg_diag)

        # Solve (X^T X + reg) W = X^T Y
        A = XtX + reg
        XtY = X_aug.t() @ Y  # [feat, out_dim]

        try:
            W_aug = torch.linalg.solve(A, XtY)
        except RuntimeError:
            # Fallback to pseudo-inverse
            W_aug = torch.pinverse(A) @ XtY

        if bias:
            W = W_aug[:-1, :].t()  # [out_dim, input_dim]
            b = W_aug[-1, :]  # [out_dim]
        else:
            W = W_aug.t()
            b = torch.zeros(W.shape[0], device=device, dtype=dtype)

        return W, b

    @staticmethod
    def solve_multi_task(
        X: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        lambdas: Dict[str, float],
        task_configs: Dict,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Solve ridge regression for multiple tasks independently.

        Args:
            X: [num_samples, reservoir_size]
            targets: dict of {task_name: Y_tensor}
            lambdas: dict of {task_name: lambda_reg}
            task_configs: config dict with task settings

        Returns:
            dict of {task_name: (W, b)}
        """
        solutions = {}
        for task_name, Y in targets.items():
            if task_name not in task_configs:
                continue
            lam = lambdas.get(task_name, 1e-6)
            bias = task_configs[task_name].get("bias", True)
            W, b = RidgeRegression.solve(X, Y, lambd=lam, bias=bias)
            solutions[task_name] = (W, b)
        return solutions


def test_ridge():
    """Test ridge regression solver."""
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(1000, 50)
    W_true = torch.randn(50, 10)
    b_true = torch.randn(10)
    Y = torch.matmul(X, W_true) + b_true + 0.1 * torch.randn(1000, 10)

    # Solve
    ridge = RidgeRegression()
    W_pred, b_pred = ridge.solve(X, Y, lambd=1e-4)

    # Check reconstruction
    Y_pred = torch.matmul(X, W_pred.t()) + b_pred
    mse = torch.mean((Y - Y_pred) ** 2)
    print(f"Reconstruction MSE: {mse.item():.6f}")
    assert mse < 0.1  # Should be small

    print("Ridge regression test passed!")


if __name__ == "__main__":
    test_ridge()
