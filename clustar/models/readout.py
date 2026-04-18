"""
Readout head and ridge regression solver for CluSTAR action classification.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ActionReadout(nn.Module):
    """
    Linear readout for action classification, trained via ridge regression.
    """

    def __init__(self, feature_dim: int, num_classes: int = 4):
        super().__init__()
        self.head = nn.Linear(feature_dim, num_classes, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, feature_dim]

        Returns:
            logits: [batch, num_classes]
        """
        return self.head(state)


class RidgeRegression:
    """
    Closed-form ridge regression solver.
    Solves: min ||Y - XW||^2 + λ||W||^2
    Solution: W = (X^T X + λI)^(-1) X^T Y
    """

    @staticmethod
    def solve(
        X: torch.Tensor, Y: torch.Tensor, lambd: float = 1e-6, bias: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve ridge regression via normal equations.

        Args:
            X: [num_samples, feature_dim]
            Y: [num_samples, out_dim] (one-hot for classification)
            lambd: regularization strength
            bias: whether to include bias term

        Returns:
            W: [out_dim, feature_dim]
            b: [out_dim]
        """
        device = X.device
        dtype = X.dtype
        lambd = float(lambd)

        if bias:
            X_aug = torch.cat(
                [X, torch.ones(X.shape[0], 1, device=device, dtype=dtype)], dim=1
            )
        else:
            X_aug = X

        feat = X_aug.shape[1]

        XtX = X_aug.t() @ X_aug
        reg_diag = torch.full((feat,), lambd, device=device, dtype=dtype)
        if bias:
            reg_diag[-1] = 0.0
        reg = torch.diag(reg_diag)

        A = XtX + reg
        XtY = X_aug.t() @ Y

        try:
            W_aug = torch.linalg.solve(A, XtY)
        except RuntimeError:
            W_aug = torch.pinverse(A) @ XtY

        if bias:
            W = W_aug[:-1, :].t()
            b = W_aug[-1, :]
        else:
            W = W_aug.t()
            b = torch.zeros(W.shape[0], device=device, dtype=dtype)

        return W, b
