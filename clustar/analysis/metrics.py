"""
Metrics Computation for CluSTAR.
Standard classification and regression metrics.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)


class MetricsCalculator:
    """
    Compute comprehensive evaluation metrics.
    """

    def __init__(self, action_names: list = None):
        self.action_names = action_names or [
            "moving",
            "spinning",
            "stationary",
        ]

    def classification_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "macro"
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: [N] true labels
            y_pred: [N] predicted labels

        Returns:
            dict with accuracy, f1, precision, recall
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        acc = accuracy_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np, average=average, zero_division=0)
        prec = precision_score(y_true_np, y_pred_np, average=average, zero_division=0)
        rec = recall_score(y_true_np, y_pred_np, average=average, zero_division=0)

        # Per-class metrics
        per_class_f1 = f1_score(y_true_np, y_pred_np, average=None, zero_division=0)
        per_class_prec = precision_score(
            y_true_np, y_pred_np, average=None, zero_division=0
        )
        per_class_rec = recall_score(
            y_true_np, y_pred_np, average=None, zero_division=0
        )

        metrics = {
            "accuracy": acc,
            "f1_macro": f1
            if average == "macro"
            else f1_score(y_true_np, y_pred_np, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true_np, y_pred_np, average="weighted", zero_division=0
            ),
            "precision_macro": prec
            if average == "macro"
            else precision_score(
                y_true_np, y_pred_np, average="macro", zero_division=0
            ),
            "recall_macro": rec
            if average == "macro"
            else recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        }

        # Per-class
        for i, cls_name in enumerate(self.action_names):
            if i < len(per_class_f1):
                metrics[f"f1_{cls_name}"] = per_class_f1[i]
                metrics[f"prec_{cls_name}"] = per_class_prec[i]
                metrics[f"rec_{cls_name}"] = per_class_rec[i]

        return metrics

    def regression_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute regression metrics (MSE, MAE, RMSE).
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mse)

        return {"mse": mse, "mae": mae, "rmse": rmse}

    def confusion_matrix_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, normalize: str = None
    ) -> np.ndarray:
        """
        Compute confusion matrix. normalize: 'true', 'pred', None
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        cm = confusion_matrix(y_true_np, y_pred_np)

        if normalize == "true":
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm.astype("float") / cm.sum(axis=0, keepdims=True)

        return cm

    def compute_all_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, task: str = "classification"
    ) -> Dict[str, float]:
        """
        Compute metrics based on task type.
        """
        if task == "classification":
            return self.classification_metrics(y_true, y_pred)
        elif task == "regression":
            return self.regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task}")


def test_metrics():
    """Test metrics computation."""
    calc = MetricsCalculator()

    # Classification test
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 0, 2])
    metrics = calc.classification_metrics(y_true, y_pred)
    print("Classification metrics:", metrics)
    assert "accuracy" in metrics

    # Regression test
    y_true_reg = torch.randn(100)
    y_pred_reg = y_true_reg + 0.1 * torch.randn(100)
    reg_metrics = calc.regression_metrics(y_true_reg, y_pred_reg)
    print("Regression metrics:", reg_metrics)
    assert "mse" in reg_metrics

    # Confusion matrix
    cm = calc.confusion_matrix_metrics(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    print("Metrics tests passed!")


if __name__ == "__main__":
    test_metrics()
