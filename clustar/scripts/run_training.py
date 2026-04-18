#!/usr/bin/env python3
"""
CluSTAR Action Classifier Training.
Trains linear action classifier on frozen reservoir states using ridge regression.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import yaml
import torch
import json
from clustar.data.dataset import get_dataloaders
from clustar.models.encoder import SpatiotemporalEncoder
from clustar.models.reservoir import ClusteredReservoir
from clustar.models.deep_reservoir import DeepReservoir
from clustar.finetune.trainer import ActionClassifier
from clustar.analysis.visualize import CluSTARVisualizer


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
    encoder = SpatiotemporalEncoder(
        canvas_size=config["data"]["canvas_size"],
        spatial_dim=config["reservoir"]["encoder"]["spatial_dim"],
        temporal_dim=config["reservoir"]["encoder"]["temporal_dim"],
        seed=config["training"]["seed"],
    )
    encoder_output_dim = (
        config["reservoir"]["encoder"]["spatial_dim"]
        + config["reservoir"]["encoder"]["temporal_dim"]
    )
    layer1 = ClusteredReservoir(
        input_dim=encoder_output_dim,
        reservoir_size=config["reservoir"]["size"],
        num_clusters=config["reservoir"]["num_clusters"],
        cluster_connectivity=config["reservoir"]["cluster"]["within_prob"],
        inter_cluster_connectivity=config["reservoir"]["cluster"]["between_prob"],
        spectral_radius_global=config["reservoir"]["spectral_radius_global"],
        input_scaling=config["reservoir"]["input_scaling"],
        bias_scaling=config["reservoir"]["bias_scaling"],
        seed=config["training"]["seed"] + 10,
        activation=config["reservoir"]["activation"],
    )

    l2_cfg = config["reservoir"]["layer2"]
    layer2 = ClusteredReservoir(
        input_dim=config["reservoir"]["size"] * 2,
        reservoir_size=l2_cfg["size"],
        num_clusters=l2_cfg["num_clusters"],
        cluster_connectivity=l2_cfg["cluster"]["within_prob"],
        inter_cluster_connectivity=l2_cfg["cluster"]["between_prob"],
        spectral_radius_global=l2_cfg["spectral_radius_global"],
        input_scaling=l2_cfg["input_scaling"],
        bias_scaling=l2_cfg["bias_scaling"],
        seed=config["training"]["seed"] + 20,
        activation=l2_cfg["activation"],
    )

    reservoir = DeepReservoir(layer1, layer2)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    feature_dim = config["reservoir"]["size"] * 4 + l2_cfg["size"] * 4
    print(f"\nModel parameters:")
    print(f"  Encoder (frozen):       {0:>12,} params")
    print(f"  Reservoir L1 (frozen):  {0:>12,} params")
    print(f"  Reservoir L2 (frozen):  {0:>12,} params")
    print(f"  Feature dim:            {feature_dim:>12,} D")

    # ==================== Training ====================
    classifier = ActionClassifier(
        reservoir=reservoir,
        encoder=encoder,
        config=config,
    )

    trainable = count_params(classifier.readout)
    print(f"  Readout (trainable):    {trainable:>12,} params")
    print()

    train_metrics = classifier.train_classifier(train_loader, val_loader)
    test_metrics = classifier.test(test_loader)

    # Save
    classifier.save_model("checkpoints/clustar.pt")

    results = {**train_metrics, **test_metrics}
    with open("logs/finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ==================== Visualization ====================
    if config["visualization"]["enabled"]:
        print("\nGenerating visualizations...")
        X_test, y_test = classifier.extract_features(test_loader)
        X_test_scaled = classifier._standardize(X_test)
        viz = CluSTARVisualizer(reservoir.layer1)
        plot_path = Path("visualization") / "plot.png"
        viz.plot_tsne_by_action(
            X_test_scaled,
            y_test,
            model_accuracy=test_metrics["test_accuracy"],
            perplexity=config["visualization"].get("tsne_perplexity", 50),
            save_path=plot_path,
        )

    print("\nFine-tuning complete!")
    print(f"Test accuracy: {test_metrics['test_accuracy']:.2%}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reservoir.yaml")
    args = parser.parse_args()
    main(args)
