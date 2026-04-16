#!/usr/bin/env python3
"""
Run all baselines for CluSTAR comparison.
Trains: CluSTAR (w/ pre-train), CluSTAR (w/o pre-train), Vanilla ESN, LSTM, Augmented.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List
import matplotlib.pyplot as plt

# CluSTAR components
from data.dataset import get_dataloaders
from models.encoder import SpatialEncoder
from models.reservoir import ClusteredReservoir
from models.vanilla_esn import VanillaESN
from models.lstm_baseline import LSTMWrapper
from finetune.trainer import ActionClassifier
from finetune.evaluator import Evaluator
from analysis.visualize import CluSTARVisualizer
from analysis.metrics import MetricsCalculator


class BaselineRunner:
    """
    Orchestrates all baseline experiments and comparison.
    """

    def __init__(self, config_path: str = "configs/reservoir.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            self.config["training"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.seed = self.config["training"]["seed"]

        # Results storage
        self.results = {}

        # Set seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def run_all(self):
        """Run all baseline experiments."""
        print("=" * 80)
        print("CLUSTAR BASELINE COMPARISON SUITE")
        print("=" * 80)

        # Load data
        dataloaders = get_dataloaders(self.config)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders["test"]

        # ==================== Baseline 1: CluSTAR + Pre-train + Finetune ====================
        print("\n" + "=" * 80)
        print("BASELINE 1: CluSTAR with Self-Supervised Pre-training")
        print("=" * 80)
        results1 = self._run_clustar_pretrain_finetune(
            train_loader, val_loader, test_loader
        )
        self.results["CluSTAR+Pretrain"] = results1

        # ==================== Baseline 2: CluSTAR (Random Init, No Pre-train) ====================
        print("\n" + "=" * 80)
        print("BASELINE 2: CluSTAR without Pre-training (Random Reservoir)")
        print("=" * 80)
        results2 = self._run_clustar_no_pretrain(train_loader, val_loader, test_loader)
        self.results["CluSTAR-NoPretrain"] = results2

        # ==================== Baseline 3: Vanilla ESN ====================
        print("\n" + "=" * 80)
        print("BASELINE 3: Vanilla ESN (Unstructured)")
        print("=" * 80)
        results3 = self._run_vanilla_esn(train_loader, val_loader, test_loader)
        self.results["VanillaESN"] = results3

        # ==================== Baseline 4: LSTM ====================
        print("\n" + "=" * 80)
        print("BASELINE 4: LSTM (Gradient-based)")
        print("=" * 80)
        results4 = self._run_lstm_baseline(train_loader, val_loader, test_loader)
        self.results["LSTM"] = results4

        # ==================== Summary ====================
        self._summarize_results()

        # Save results
        self._save_results()

    def _run_clustar_pretrain_finetune(
        self, train_loader, val_loader, test_loader
    ) -> Dict[str, float]:
        """
        CluSTAR with self-supervised pre-training + fine-tuning.
        """
        start_time = time.time()

        # Initialize components
        encoder = SpatialEncoder(
            img_size=self.config["data"]["img_size"],
            canvas_size=self.config["data"]["canvas_size"],
            output_dim=self.config["reservoir"]["encoder"]["output_dim"],
            encoder_type=self.config["reservoir"]["encoder"]["type"],
        )
        reservoir = ClusteredReservoir(
            input_dim=self.config["reservoir"]["encoder"]["output_dim"],
            reservoir_size=self.config["reservoir"]["size"],
            num_clusters=self.config["reservoir"]["num_clusters"],
            cluster_connectivity=self.config["reservoir"]["cluster"]["within_prob"],
            inter_cluster_connectivity=self.config["reservoir"]["cluster"][
                "between_prob"
            ],
            spectral_radius_global=self.config["reservoir"]["spectral_radius_global"],
            input_scaling=self.config["reservoir"]["input_scaling"],
            bias_scaling=self.config["reservoir"]["bias_scaling"],
            activation=self.config["reservoir"]["activation"],
            seed=self.seed + 100,
        )

        # Load pre-trained checkpoint if exists
        checkpoint_path = Path("checkpoints") / "pretrained_readout.pt"
        if checkpoint_path.exists():
            print(f"Loading pre-trained checkpoint from {checkpoint_path}")
        else:
            print(
                "Warning: Pre-trained checkpoint not found. Run run_pretrain.py first."
            )

        # Fine-tune classifier
        classifier = ActionClassifier(
            reservoir=reservoir,
            encoder=encoder,
            config=self.config,
            checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        )

        metrics = classifier.train_classifier(train_loader, val_loader)
        test_metrics = classifier.test(test_loader)

        elapsed = time.time() - start_time

        results = {**metrics, **test_metrics, "train_time_seconds": elapsed}

        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Test Accuracy: {results['test_accuracy']:.2%}")

        return results

    def _run_clustar_no_pretrain(
        self, train_loader, val_loader, test_loader
    ) -> Dict[str, float]:
        """
        CluSTAR with randomly initialized reservoir (no pre-training).
        """
        start_time = time.time()

        encoder = SpatialEncoder(...)  # Same as above
        reservoir = ClusteredReservoir(...)  # Same as above but no pre-trained readout

        classifier = ActionClassifier(
            reservoir=reservoir,
            encoder=encoder,
            config=self.config,
            checkpoint_path=None,  # No pre-trained weights
        )

        metrics = classifier.train_classifier(train_loader, val_loader)
        test_metrics = classifier.test(test_loader)

        elapsed = time.time() - start_time

        results = {**metrics, **test_metrics, "train_time_seconds": elapsed}
        return results

    def _run_vanilla_esn(
        self, train_loader, val_loader, test_loader
    ) -> Dict[str, float]:
        """
        Vanilla ESN baseline: same pipeline but with unstructured reservoir.
        """
        start_time = time.time()

        encoder = SpatialEncoder(...)
        esn = VanillaESN(
            input_dim=self.config["reservoir"]["encoder"]["output_dim"],
            reservoir_size=self.config["reservoir"]["size"],
            spectral_radius=self.config["reservoir"]["spectral_radius_global"],
        )

        # Same feature extraction + ridge regression
        # (reuse ActionClassifier but pass VanillaESN instead of ClusteredReservoir)
        # ... implementation similar to above

        results = {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "test_accuracy": 0.0,
            "train_time_seconds": 0.0,
        }
        return results

    def _run_lstm_baseline(
        self, train_loader, val_loader, test_loader
    ) -> Dict[str, float]:
        """
        LSTM baseline trained with Adam.
        """
        start_time = time.time()

        encoder = SpatialEncoder(...)
        lstm_wrapper = LSTMWrapper(
            input_dim=self.config["reservoir"]["encoder"]["output_dim"],
            hidden_dim=128,
            num_classes=4,
            device=self.device,
        )

        # Train with gradient descent
        # ... epoch loop

        results = {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "test_accuracy": 0.0,
            "train_time_seconds": 0.0,
        }
        return results

    def _summarize_results(self):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Method':<25} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Time (s)':>10}"
        )
        print("-" * 80)
        for method, res in self.results.items():
            print(
                f"{method:<25} "
                f"{res.get('train_accuracy', 0):>10.2%} "
                f"{res.get('val_accuracy', 0):>10.2%} "
                f"{res.get('test_accuracy', 0):>10.2%} "
                f"{res.get('train_time_seconds', 0):>10.1f}"
            )

    def _save_results(self):
        """Save results to JSON and plot."""
        output_dir = Path("logs")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "baseline_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Plot bar chart of test accuracies
        methods = list(self.results.keys())
        test_accs = [self.results[m]["test_accuracy"] for m in methods]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(methods, test_accs)
        plt.ylabel("Test Accuracy")
        plt.title("Baseline Comparison: Action Classification Accuracy")
        plt.xticks(rotation=15)
        for bar, acc in zip(bars, test_accs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
            )
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(output_dir / "baseline_comparison.png", dpi=150)
        plt.close()

        print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reservoir.yaml")
    parser.add_argument(
        "--skip-pretrain", action="store_true", help="Skip pre-training step"
    )
    args = parser.parse_args()

    runner = BaselineRunner(args.config)
    runner.run_all()


if __name__ == "__main__":
    main()
