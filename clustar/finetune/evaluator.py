"""
Evaluation Metrics for CluSTAR.
Computes accuracy, F1, confusion matrix, and world model metrics (MSE).
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


class Evaluator:
    """
    Comprehensive evaluation for action classification and world model.
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.action_names = ["moving", "spinning", "collision", "stationary"]

    def evaluate_classification(
        self,
        dataloader: DataLoader,
        reservoir: torch.nn.Module,
        encoder: torch.nn.Module,
        readout: torch.nn.Module,
        aggregation: str = "mean",
    ) -> Dict[str, float]:
        """
        Evaluate action classification accuracy.

        Returns:
            metrics dict with accuracy, f1, etc.
        """
        reservoir.eval()
        encoder.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                frames = batch["frames"].to(self.device)
                labels = batch["label"].numpy()

                B, T = frames.shape[0], frames.shape[1]
                flat_frames = frames.view(B * T, 1, 64, 64)
                encoded = encoder(flat_frames).view(B, T, -1)
                states, _ = reservoir.forward_sequence(encoded)

                # Aggregate states
                if aggregation == "mean":
                    feats = states.mean(dim=1)
                elif aggregation == "last":
                    feats = states[:, -1, :]
                elif aggregation == "cat_last3":
                    feats = states[:, -3:, :].reshape(B, -1)

                logits = readout(feats, "action_classification")
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        cm = confusion_matrix(all_labels, all_preds)

        # Per-class accuracy
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
        print(f"\nPer-class accuracy:")
        for i, cls_name in enumerate(self.action_names):
            print(f"  {cls_name}: {per_class_acc[i]:.2%}")

        return metrics, cm

    def evaluate_frame_prediction(
        self,
        dataloader: DataLoader,
        reservoir: torch.nn.Module,
        encoder: torch.nn.Module,
        readout: torch.nn.Module,
        horizon: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate world model quality: predict next frame.

        Args:
            horizon: how many steps to predict ahead (1 = next frame)

        Returns:
            MSE, PSNR, SSIM (if available)
        """
        reservoir.eval()
        encoder.eval()
        total_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                frames = batch["frames"].to(self.device)  # [B, T, 1, H, W]
                B, T = frames.shape[0], frames.shape[1]

                for b in range(B):
                    # Run reservoir on sequence
                    seq_frames = frames[b]  # [T, 1, H, W]
                    flat = seq_frames.view(T, 1, 64, 64)
                    encoded = encoder(flat).unsqueeze(0)  # [1, T, D]
                    states, _ = reservoir.forward_sequence(encoded)  # [1, T, N]

                    # Predict frames autoregressively
                    for t in range(T - horizon):
                        state_t = states[0, t]  # [N]
                        pred_frame = readout.forward(
                            state_t.unsqueeze(0), "frame_prediction"
                        )  # [1, 784]
                        pred_frame = pred_frame.view(1, 1, 64, 64)

                        target_frame = frames[b, t + horizon]  # [1, H, W]

                        mse = torch.mean((pred_frame - target_frame.unsqueeze(0)) ** 2)
                        total_mse += mse.item()
                        total_samples += 1

        avg_mse = total_mse / total_samples
        print(f"\nFrame prediction MSE (h={horizon}): {avg_mse:.6f}")
        return {"frame_mse": avg_mse}

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

    def plot_tsne(
        self,
        states: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[str] = None,
    ):
        """
        Plot t-SNE visualization of reservoir states colored by action.
        """
        from sklearn.manifold import TSNE

        states_np = states.cpu().numpy()
        labels_np = labels.cpu().numpy()

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        states_2d = tsne.fit_transform(states_np)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            states_2d[:, 0], states_2d[:, 1], c=labels_np, cmap="tab10", alpha=0.6, s=10
        )
        plt.legend(handles=scatter.legend_elements()[0], labels=self.action_names)
        plt.title("t-SNE of Reservoir States (colored by action)")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


class OnlineEvaluator:
    """
    Track metrics during training (train/val curves).
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_accuracy": [],
            "val_accuracy": [],
            "train_loss": [],
            "val_loss": [],
        }

    def log(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def plot_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        axes[0].plot(self.history["train_accuracy"], label="Train")
        axes[0].plot(self.history["val_accuracy"], label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy over Training")
        axes[0].legend()
        axes[0].grid(True)

        # Loss
        axes[1].plot(self.history["train_loss"], label="Train")
        axes[1].plot(self.history["val_loss"], label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss over Training")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_history(self, path: Optional[str] = None):
        """Save training history as JSON."""
        if path is None:
            path = self.log_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def test_evaluator():
    """Quick test of evaluator."""
    from models.reservoir import ClusteredReservoir
    from models.encoder import SpatialEncoder
    from models.readout import MultiTaskReadout

    config = {
        "reservoir": {
            "encoder": {"output_dim": 128},
            "size": 100,
            "num_clusters": 5,
            "cluster": {"within_prob": 0.3, "between_prob": 0.02},
            "spectral_radius_global": 0.95,
        },
        "data": {"canvas_size": 64, "img_size": 28},
        "training": {"device": "cpu"},
    }

    reservoir = ClusteredReservoir(input_dim=128, reservoir_size=100, num_clusters=5)
    encoder = SpatialEncoder(output_dim=128)
    readout = MultiTaskReadout(
        reservoir_size=100, task_configs={"action_classification": {"enabled": True}}
    )

    # Dummy dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            return {
                "frames": torch.rand(30, 1, 64, 64),
                "label": torch.randint(0, 4, ()).long(),
            }

    loader = DataLoader(DummyDataset(), batch_size=4)

    evaluator = Evaluator(config)
    metrics = evaluator.evaluate_classification(loader, reservoir, encoder, readout)
    print("Metrics:", metrics)

    # Test frame prediction
    metrics2 = evaluator.evaluate_frame_prediction(loader, reservoir, encoder, readout)
    print("World model:", metrics2)

    print("Evaluator tests passed!")


if __name__ == "__main__":
    test_evaluator()
