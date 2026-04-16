#!/usr/bin/env python3
"""
Main script for CluSTAR self-supervised pre-training.
Runs reservoir state collection and multi-task ridge regression training.
"""

import argparse
import yaml
import torch
from pathlib import Path
from data.dataset import get_pretrain_dataloader
from models.reservoir import ClusteredReservoir
from models.encoder import SpatialEncoder
from pretrain.trainer import MultiTaskTrainer


def main(args):
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set random seeds
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    # Device
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ==================== 1. Data ====================
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    pretrain_loader = get_pretrain_dataloader(config)
    print(f"Pre-train dataloader: {len(pretrain_loader)} batches")

    # ==================== 2. Model Components ====================
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    # Spatial encoder
    encoder = SpatialEncoder(
        img_size=config["data"]["img_size"],
        canvas_size=config["data"]["canvas_size"],
        output_dim=config["reservoir"]["encoder"]["output_dim"],
        encoder_type=config["reservoir"]["encoder"]["type"],
        seed=config["training"]["seed"],
    )
    print(f"Spatial encoder: {config['reservoir']['encoder']['output_dim']} features")

    # Structured reservoir
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
        seed=config["training"]["seed"],
        routing_config=config["reservoir"]["routing"]
        if config["reservoir"]["routing"]["enabled"]
        else None,
    )
    print(
        f"Reservoir: {config['reservoir']['size']} neurons, {config['reservoir']['num_clusters']} clusters"
    )
    print(f"  Timescales: 3 groups (fast/medium/slow)")
    print(
        f"  Connectivity: within={config['reservoir']['cluster']['within_prob']}, between={config['reservoir']['cluster']['between_prob']}"
    )

    # ==================== 3. Pre-training ====================
    print("\n" + "=" * 60)
    print("PRE-TRAINING")
    print("=" * 60)

    trainer = MultiTaskTrainer(
        reservoir=reservoir, spatial_encoder=encoder, config=config, device=device
    )

    # Run pre-training
    readout = trainer.run(pretrain_loader, save_checkpoint=True)

    # ==================== 4. Summary ====================
    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"\nCheckpoint saved to: {config['logging']['checkpoint_dir']}/pretrained_readout.pt"
    )
    print("\nNext step: Run fine-tuning with:")
    print(
        f"  python scripts/run_finetune.py --config {args.config} --checkpoint checkpoints/pretrained_readout.pt"
    )

    return readout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CluSTAR Self-Supervised Pre-training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reservoir.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to resume from",
    )
    args = parser.parse_args()

    main(args)
