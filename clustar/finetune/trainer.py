"""
Downstream Fine-tuning for Action Classification.
Uses pre-trained CluSTAR reservoir and trains linear classifier via ridge regression.
"""

import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from models.reservoir import ClusteredReservoir
from models.encoder import SpatialEncoder
from models.readout import RidgeRegression, MultiTaskReadout
from data.dataset import get_dataloaders
import json


class ActionClassifier:
    """
    Fine-tunes a linear classifier on top of frozen CluSTAR reservoir states.
    """

    def __init__(
        self,
        reservoir: ClusteredReservoir,
        encoder: SpatialEncoder,
        config: Dict,
        checkpoint_path: Optional[str] = None,
    ):
        self.reservoir = reservoir
        self.encoder = encoder
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # Freeze reservoir and encoder
        self.reservoir.eval().to(self.device)
        for param in self.reservoir.parameters():
            param.requires_grad = False
        self.encoder.eval().to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Readout (classifier)
        self.readout = MultiTaskReadout(
            reservoir_size=config["reservoir"]["size"],
            task_configs={"action_classification": {"enabled": True}},
            device=self.device,
        ).to(self.device)

        # Load pre-trained weights if available
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

        # Metrics tracking
        self.train_acc = []
        self.val_acc = []

    def _load_pretrained(self, path: str):
        """Load pre-trained readout weights (optional)."""
        ckpt = torch.load(path, map_location=self.device)
        # Only load if task exists in checkpoint
        if "frame_prediction" in ckpt.get("solutions", {}):
            print("Note: Pre-trained readout found but not used for initialization")
            # Could initialize from pre-trained features if desired

    @torch.no_grad()
    def extract_features(
        self, dataloader: DataLoader, aggregation: str = "mean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run reservoir over dataset and extract state features.

        Args:
            dataloader: labeled data
            aggregation: how to pool temporal dimension: 'mean', 'last', 'cat_last3'

        Returns:
            X: [num_samples, reservoir_size] features
            y: [num_samples] labels
        """
        features = []
        labels = []

        print(f"Extracting features from {len(dataloader)} batches...")
        for batch in dataloader:
            frames = batch["frames"].to(self.device)  # [B, T, 1, H, W]
            labels_batch = batch["label"].to(self.device)

            B, T = frames.shape[0], frames.shape[1]
            flat_frames = frames.view(B * T, 1, 64, 64)
            encoded = self.encoder(flat_frames).view(B, T, -1)  # [B, T, D]

            states, _ = self.reservoir.forward_sequence(encoded)  # [B, T, N]

            # Temporal aggregation
            if aggregation == "mean":
                feats = states.mean(dim=1)  # [B, N]
            elif aggregation == "last":
                feats = states[:, -1, :]  # [B, N]
            elif aggregation == "cat_last3":
                feats = states[:, -3:, :].reshape(B, -1)  # [B, 3N]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            features.append(feats.cpu())
            labels.append(labels_batch.cpu())

        X = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)

        print(f"Extracted features: {X.shape}, labels: {y.shape}")
        return X, y

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ridge_lambda: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train linear classifier via ridge regression.

        Args:
            train_loader: training data
            val_loader: validation data
            ridge_lambda: regularization strength (from config if None)

        Returns:
            metrics dict with train/val accuracy
        """
        if ridge_lambda is None:
            ridge_lambda = self.config["finetune"]["ridge_lambda"]

        print("\n" + "=" * 60)
        print("DOWNSTREAM FINE-TUNING")
        print("=" * 60)

        # Extract features from train and val
        X_train, y_train = self.extract_features(train_loader)
        X_val, y_val = self.extract_features(val_loader)

        print(
            f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples"
        )

        # Convert labels to one-hot for ridge regression
        num_classes = self.config["finetune"]["num_classes"]
        y_train_onehot = torch.zeros(len(y_train), num_classes, device=self.device)
        y_train_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)

        # Train ridge regression
        print(f"Solving ridge regression (λ={ridge_lambda})...")
        ridge = RidgeRegression()
        W, b = ridge.solve(X_train, y_train_onehot, lambd=ridge_lambda, bias=True)

        # Set readout weights
        self.readout.heads["action_classification"].weight.data = W
        self.readout.heads["action_classification"].bias.data = b

        # Evaluate
        train_acc = self._evaluate(X_train, y_train)
        val_acc = self._evaluate(X_val, y_val)

        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Val accuracy:   {val_acc:.2%}")

        return {"train_accuracy": train_acc, "val_accuracy": val_acc}

    @torch.no_grad()
    def _evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy."""
        logits = self.readout.forward(X.to(self.device), "action_classification")
        preds = torch.argmax(logits, dim=1)
        correct = (preds.cpu() == y).float().mean()
        return correct.item()

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set."""
        X_test, y_test = self.extract_features(test_loader)
        test_acc = self._evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_acc:.2%}")
        return {"test_accuracy": test_acc}

    def save_model(self, path: str):
        """Save fine-tuned classifier."""
        save_dict = {
            "readout_state_dict": self.readout.state_dict(),
            "config": self.config,
            "metrics": {
                "train_acc": self.train_acc[-1] if self.train_acc else None,
                "val_acc": self.val_acc[-1] if self.val_acc else None,
            },
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load fine-tuned classifier."""
        ckpt = torch.load(path, map_location=self.device)
        self.readout.load_state_dict(ckpt["readout_state_dict"])
        print(f"Model loaded from {path}")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # ==================== Data ====================
    print("Loading datasets...")
    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # ==================== Model ====================
    print("\nInitializing CluSTAR components...")
    encoder = SpatialEncoder(
        img_size=config["data"]["img_size"],
        canvas_size=config["data"]["canvas_size"],
        output_dim=config["reservoir"]["encoder"]["output_dim"],
        encoder_type=config["reservoir"]["encoder"]["type"],
    )
    reservoir = ClusteredReservoir(
        input_dim=config["reservoir"]["encoder"]["output_dim"],
        reservoir_size=config["reservoir"]["size"],
        num_clusters=config["reservoir"]["num_clusters"],
        cluster_connectivity=config["reservoir"]["cluster"]["within_prob"],
        inter_cluster_connectivity=config["reservoir"]["cluster"]["between_prob"],
        spectral_radius_global=config["reservoir"]["spectral_radius_global"],
        seed=config["training"]["seed"],
    )

    # ==================== Fine-tuning ====================
    classifier = ActionClassifier(
        reservoir=reservoir,
        encoder=encoder,
        config=config,
        checkpoint_path=args.checkpoint,
    )

    # Train classifier
    metrics = classifier.train_classifier(train_loader, val_loader)

    # Test
    test_metrics = classifier.test(test_loader)

    # Save
    classifier.save_model("checkpoints/finetuned_classifier.pt")

    # Save metrics
    with open("logs/finetune_metrics.json", "w") as f:
        json.dump({**metrics, **test_metrics}, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CluSTAR Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/reservoir.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Pre-trained checkpoint path"
    )
    args = parser.parse_args()

    main(args)
