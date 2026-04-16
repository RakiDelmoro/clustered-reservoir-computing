#!/usr/bin/env python3
"""
CluSTAR Fine-tuning Script.
Trains linear action classifier on frozen reservoir states using ridge regression.
"""

import argparse
import yaml
import torch
from pathlib import Path
from data.dataset import get_dataloaders
from models.encoder import SpatialEncoder
from models.reservoir import ClusteredReservoir
from finetune.trainer import ActionClassifier
from analysis.metrics import MetricsCalculator
import json


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ==================== Data ====================
    print("Loading datasets...")
    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # ==================== Model ====================
    print("Initializing CluSTAR components...")
    encoder = SpatialEncoder(
        img_size=config["data"]["img_size"],
        canvas_size=config["data"]["canvas_size"],
        output_dim=config["reservoir"]["encoder"]["output_dim"],
        encoder_type=config["reservoir"]["encoder"]["type"],
        seed=config["training"]["seed"],
    )
    reservoir = ClusteredReservoir(
        input_dim=config["reservoir"]["encoder"]["output_dim"],
        reservoir_size=config["reservoir"]["size"],
        num_clusters=config["reservoir"]["num_clusters"],
        cluster_connectivity=config["reservoir"]["cluster"]["within_prob"],
        inter_cluster_connectivity=config["reservoir"]["cluster"]["between_prob"],
        spectral_radius_global=config["reservoir"]["spectral_radius_global"],
        input_scaling=config["reservoir"]["input_scaling"],
        bias_scaling=config["reservoir"]["bias_scaling"],
        activation=config["reservoir"]["activation"],
        seed=config["training"]["seed"] + 10,
        routing_config=config["reservoir"]["routing"]
        if config["reservoir"]["routing"]["enabled"]
        else None,
    )

    # ==================== Fine-tuning ====================
    classifier = ActionClassifier(
        reservoir=reservoir,
        encoder=encoder,
        config=config,
        checkpoint_path=args.checkpoint,
    )

    # Train (ridge regression)
    train_metrics = classifier.train_classifier(train_loader, val_loader)
    test_metrics = classifier.test(test_loader)

    # Save everything
    classifier.save_model("checkpoints/finetuned_classifier.pt")

    results = {**train_metrics, **test_metrics}
    with open("logs/finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ==================== Visualizations (optional) ====================
    if config["visualization"]["enabled"]:
        print("\nGenerating visualizations...")
        # Extract states on test set
        X_test, y_test = classifier.extract_features(test_loader)
        viz = CluSTARVisualizer(reservoir)
        output_dir = Path("visualizations") / "finetune"
        output_dir.mkdir(parents=True, exist_ok=True)
        viz.plot_tsne_by_action(X_test, y_test, save_path=output_dir / "tsne_test.png")
        print(f"  Saved to {output_dir}/")

    print("\nFine-tuning complete!")
    print(f"Test accuracy: {test_metrics['test_accuracy']:.2%}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reservoir.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Pre-trained checkpoint (optional)"
    )
    args = parser.parse_args()

    main(args)
