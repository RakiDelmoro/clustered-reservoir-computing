"""
Multi-Task Ridge Regression Trainer for CluSTAR Pre-training.
Trains all readout heads jointly via closed-form ridge regression.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from .collect_states import ReservoirStateCollector
from .tasks import PretextTaskFactory
from models.readout import MultiTaskReadout, RidgeRegression


class MultiTaskTrainer:
    """
    Orchestrates self-supervised pre-training:
    1. Collect reservoir states from unlabeled data
    2. Compute pretext task targets
    3. Solve ridge regression for all heads jointly or independently
    4. Save pre-trained checkpoint
    """

    def __init__(self, reservoir, spatial_encoder, config: Dict, device: str = "cpu"):
        self.reservoir = reservoir
        self.spatial_encoder = spatial_encoder
        self.config = config
        self.device = device

        # Components
        self.collector = ReservoirStateCollector(reservoir, spatial_encoder, device)
        self.task_factory = PretextTaskFactory(config, device)
        self.readout = MultiTaskReadout(
            reservoir_size=config["reservoir"]["size"],
            task_configs=config["pretrain"]["tasks"],
            device=device,
        )

        # Output dirs
        self.checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self, dataloader, save_checkpoint: bool = True) -> MultiTaskReadout:
        """
        Full pre-training pipeline.

        Args:
            dataloader: unlabeled data loader
            save_checkpoint: whether to save trained readout

        Returns:
            trained readout module (with fixed reservoir)
        """
        print("=" * 60)
        print("CLUSTAR SELF-SUPERVISED PRE-TRAINING")
        print("=" * 60)

        # Step 1: Collect states and compute targets
        collected = self.collector.collect_with_targets(dataloader, self.task_factory)
        X_dict = collected["X"]  # Dict: task -> [num_samples, N]
        targets = collected["targets"]  # Dict: task -> targets

        print("\nCollected data shapes:")
        for task, X_task in X_dict.items():
            tgt = targets[task]
            if isinstance(tgt, dict):
                print(
                    f"  {task}: X={X_task.shape}, targets={ {k: v.shape for k, v in tgt.items()} }"
                )
            else:
                print(f"  {task}: X={X_task.shape}, Y={tgt.shape}")

        # Step 2: Solve ridge regression for each task (handles special cases)
        print("\nTraining readout heads via ridge regression...")
        ridge = RidgeRegression()
        lambdas = {
            task: self.config["pretrain"]["ridge_lambda"]
            for task in self.config["pretrain"]["tasks"]
        }

        # Use custom solver that handles special target formats (temporal_order, rotation_contrast)
        solutions = self._solve_all_tasks(X_dict, targets, lambdas)

        # Step 3: Set readout weights
        for task, (W, b) in solutions.items():
            self.readout.set_weight_matrix(task, W, b)
            print(f"  Task '{task}': W={W.shape}, b={b.shape}")

        # Step 4: Save checkpoint
        if save_checkpoint:
            self._save_checkpoint(solutions)

        print("\nPre-training complete!")
        return self.readout

    def _solve_all_tasks(
        self,
        X_dict: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lambdas: Dict[str, float],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Solve ridge regression for each task (handling special cases).
        """
        solutions = {}

        for task_name in self.config["pretrain"]["tasks"]:
            if not self.config["pretrain"]["tasks"][task_name].get("enabled", False):
                continue

            X = X_dict[task_name]
            Y = targets[task_name]
            lam = lambdas.get(task_name, 1e-6)

            if task_name == "frame_prediction":
                # Y is [num_samples, 784]
                W, b = RidgeRegression.solve(X, Y, lambd=lam, bias=True)

            elif task_name == "temporal_order":
                # Y is dict with pair_indices_flat and labels
                pair_indices = Y["pair_indices_flat"]  # [num_pairs, 2]
                labels = Y["labels"].unsqueeze(1)  # [num_pairs, 1]
                # Build design: difference of states
                x_i = X[pair_indices[:, 0]]  # [num_pairs, N]
                x_j = X[pair_indices[:, 1]]
                X_task = x_i - x_j  # Use difference
                W, b = RidgeRegression.solve(X_task, labels, lambd=lam, bias=True)

            elif task_name == "speed_regression":
                # Y is [num_samples, 1]
                W, b = RidgeRegression.solve(X, Y, lambd=lam, bias=True)

            elif task_name == "segmentation":
                # Y is [num_samples, 784] (masks)
                W, b = RidgeRegression.solve(X, Y, lambd=lam, bias=True)

            elif task_name == "rotation_contrast":
                # Contrastive: use anchor states only
                anchor_idx = Y["anchor_idx"]
                X_anchor = X[anchor_idx]
                pseudo_labels = torch.norm(X_anchor, dim=1, keepdim=True)
                W, b = RidgeRegression.solve(
                    X_anchor, pseudo_labels, lambd=lam, bias=True
                )

            else:
                continue

            solutions[task_name] = (W, b)

        return solutions

    def _save_checkpoint(self, solutions: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Save pre-trained readout weights."""
        save_dict = {
            "config": self.config,
            "readout_state_dict": self.readout.state_dict(),
            "solutions": {
                k: {"W": v[0].cpu(), "b": v[1].cpu()} for k, v in solutions.items()
            },
        }
        path = self.checkpoint_dir / "pretrained_readout.pt"
        torch.save(save_dict, path)
        print(f"\nCheckpoint saved to {path}")

    def load_checkpoint(self, path: Optional[str] = None):
        """Load pre-trained readout."""
        if path is None:
            path = self.checkpoint_dir / "pretrained_readout.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.readout.load_state_dict(ckpt["readout_state_dict"])
        print(f"Loaded checkpoint from {path}")
        return self.readout


def test_trainer():
    """Quick integration test."""
    import yaml
    from torch.utils.data import DataLoader

    with open("configs/reservoir.yaml") as f:
        config = yaml.safe_load(f)

    # Dummy unlabeled dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {"frames": torch.rand(30, 1, 64, 64)}

    loader = DataLoader(DummyDataset(), batch_size=10, shuffle=False)

    # Initialize small components for test
    config["reservoir"]["size"] = 100
    config["reservoir"]["num_clusters"] = 5

    from models.reservoir import ClusteredReservoir
    from models.encoder import SpatialEncoder

    reservoir = ClusteredReservoir(
        input_dim=config["reservoir"]["encoder"]["output_dim"],
        reservoir_size=100,
        num_clusters=5,
        seed=42,
    )
    encoder = SpatialEncoder(output_dim=128)

    trainer = MultiTaskTrainer(reservoir, encoder, config, device="cpu")
    readout = trainer.run(loader, save_checkpoint=False)

    print("Trainer test passed!")


if __name__ == "__main__":
    test_trainer()
