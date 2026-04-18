"""
Action classification via ridge regression on frozen CluSTAR reservoir states.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from clustar.models.reservoir import ClusteredReservoir
from clustar.models.deep_reservoir import DeepReservoir
from clustar.models.encoder import SpatiotemporalEncoder
from clustar.models.readout import RidgeRegression, ActionReadout


class ActionClassifier:
    """
    Trains linear classifier on top of frozen DeepReservoir states.
    Dual-layer per-cluster temporal aggregation + feature standardization + ridge regression.
    """

    def __init__(
        self,
        reservoir: DeepReservoir,
        encoder: SpatiotemporalEncoder,
        config: Dict,
    ):
        self.reservoir = reservoir
        self.encoder = encoder
        self.config = config
        self.device = torch.device(config["training"]["device"])

        self.reservoir.eval().to(self.device)
        for param in self.reservoir.parameters():
            param.requires_grad = False
        self.encoder.eval().to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        l1_size = config["reservoir"]["size"]
        l2_size = config["reservoir"]["layer2"]["size"]
        feature_dim = l1_size * 4 + l2_size * 4

        self.readout = ActionReadout(
            feature_dim=feature_dim,
            num_classes=config["finetune"]["num_classes"],
        ).to(self.device)

    @torch.no_grad()
    def extract_features(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run deep reservoir over dataset and extract dual-layer aggregated features.

        Returns:
            X: [num_samples, 4*L1 + 4*L2] features
            y: [num_samples] labels
        """
        features = []
        labels = []

        print(f"Extracting features from {len(dataloader)} batches...")
        for batch in tqdm(
            dataloader, desc="Extracting features", total=len(dataloader)
        ):
            frames = batch["frames"].to(self.device)
            labels_batch = batch["label"].to(self.device)

            B, T = frames.shape[0], frames.shape[1]
            encoded = self.encoder(frames)

            states_L1, states_L2, _, _ = self.reservoir.forward_sequence(encoded)

            cluster_assignments_L1 = self.reservoir.layer1.cluster_assignments
            num_clusters_L1 = self.reservoir.layer1.num_clusters
            cluster_feats = []
            for c in range(num_clusters_L1):
                mask = cluster_assignments_L1 == c
                c_states = states_L1[:, :, mask]
                c0 = c_states[:, 0, :]
                cT = c_states[:, -1, :]
                c_mean = c_states.mean(dim=1)
                c_max = c_states.max(dim=1)[0]
                cluster_feats.extend([c0, cT, c_mean, c_max])

            cluster_assignments_L2 = self.reservoir.layer2.cluster_assignments
            num_clusters_L2 = self.reservoir.layer2.num_clusters
            for c in range(num_clusters_L2):
                mask = cluster_assignments_L2 == c
                c_states = states_L2[:, :, mask]
                c0 = c_states[:, 0, :]
                cT = c_states[:, -1, :]
                c_mean = c_states.mean(dim=1)
                c_max = c_states.max(dim=1)[0]
                cluster_feats.extend([c0, cT, c_mean, c_max])

            feats = torch.cat(cluster_feats, dim=1)

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
    ) -> Dict[str, float]:
        """
        Train linear classifier via ridge regression with feature standardization.

        Returns:
            metrics dict with train/val accuracy
        """
        ridge_lambda = self.config["finetune"]["ridge_lambda"]

        print("\n" + "=" * 60)
        print("DOWNSTREAM FINE-TUNING")
        print("=" * 60)

        X_train, y_train = self.extract_features(train_loader)
        X_val, y_val = self.extract_features(val_loader)

        print(
            f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples"
        )

        # Standardize features
        feat_mean = X_train.mean(dim=0, keepdim=True)
        feat_std = X_train.std(dim=0, keepdim=True).clamp(min=1e-8)
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        X_train_scaled = (X_train - feat_mean) / feat_std
        X_val_scaled = (X_val - feat_mean) / feat_std

        # Convert labels to one-hot
        num_classes = self.config["finetune"]["num_classes"]
        y_train_cpu = y_train.to("cpu")
        y_train_onehot = torch.zeros(len(y_train_cpu), num_classes, device="cpu")
        y_train_onehot.scatter_(1, y_train_cpu.unsqueeze(1), 1.0)

        # Solve ridge regression
        print(f"Solving ridge regression (λ={ridge_lambda})...")
        W, b = RidgeRegression.solve(
            X_train_scaled, y_train_onehot, lambd=ridge_lambda, bias=True
        )

        self.readout.head.weight.data = W.to(self.device)
        self.readout.head.bias.data = b.to(self.device)

        train_acc = self._evaluate(X_train_scaled, y_train)
        val_acc = self._evaluate(X_val_scaled, y_val)

        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Val accuracy:   {val_acc:.2%}")

        return {"train_accuracy": train_acc, "val_accuracy": val_acc}

    @torch.no_grad()
    def _evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy. X should already be standardized."""
        logits = self.readout.forward(X.to(self.device))
        preds = torch.argmax(logits, dim=1)
        correct = (preds.cpu() == y).float().mean()
        return correct.item()

    def _standardize(self, X: torch.Tensor) -> torch.Tensor:
        """Standardize features using stored mean/std from training set."""
        if not hasattr(self, "feat_mean"):
            return X
        return (X - self.feat_mean) / self.feat_std

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set."""
        X_test, y_test = self.extract_features(test_loader)
        X_test_scaled = self._standardize(X_test)
        test_acc = self._evaluate(X_test_scaled, y_test)
        print(f"\nTest accuracy: {test_acc:.2%}")
        return {"test_accuracy": test_acc}

    def save_model(self, path: str):
        """Save fine-tuned classifier with standardization parameters."""
        save_dict = {
            "readout_state_dict": self.readout.state_dict(),
            "config": self.config,
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load fine-tuned classifier."""
        ckpt = torch.load(path, map_location=self.device)
        self.readout.load_state_dict(ckpt["readout_state_dict"])
        if "feat_mean" in ckpt:
            self.feat_mean = ckpt["feat_mean"]
            self.feat_std = ckpt["feat_std"]
        print(f"Model loaded from {path}")
