"""
Evaluation Metrics for CluSTAR.
Computes accuracy, F1, confusion matrix.
"""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


class Evaluator:
    """
    Comprehensive evaluation for action classification.
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.action_names = ["moving", "spinning", "stationary"]

    def evaluate_classification(
        self,
        dataloader: DataLoader,
        classifier: torch.nn.Module,
    ) -> Dict[str, float]:
        """
        Evaluate action classification using a trained ActionClassifier.

        Returns:
            metrics dict and confusion matrix
        """
        X_test, y_test = classifier.extract_features(dataloader)
        X_test_scaled = classifier._standardize(X_test)
        logits = classifier.readout.forward(X_test_scaled.to(self.device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = y_test.numpy()

        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")
        cm = confusion_matrix(labels, preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class_accuracy": per_class_acc.tolist(),
        }

        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        for i, cls_name in enumerate(self.action_names):
            print(f"  {cls_name}: {per_class_acc[i]:.2%}")

        return metrics, cm

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.action_names,
            yticklabels=self.action_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Action Classification Confusion Matrix")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
