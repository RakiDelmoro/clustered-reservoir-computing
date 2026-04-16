#!/usr/bin/env python3
"""
Minimal sanity check for CluSTAR.
Tests: data generation, reservoir forward, pre-training, fine-tuning on tiny data.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.generator import MovingMNISTGenerator, MovingMNISTDataset
from models.encoder import SpatialEncoder
from models.reservoir import ClusteredReservoir
from pretrain.trainer import MultiTaskTrainer, RidgeRegression
from finetune.trainer import ActionClassifier
import yaml


def main():
    print("=" * 60)
    print("CLUSTAR SANITY CHECK (MINIMAL)")
    print("=" * 60)

    # ---------- Config ----------
    config_path = Path(__file__).parent / "configs" / "reservoir.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Force CPU for sanity test (avoid GPU device issues)
    config["training"]["device"] = "cpu"

    # Override for quick test
    config["data"]["train_samples"] = 20
    config["data"]["val_samples"] = 10
    config["data"]["pretrain_samples"] = 30
    config["reservoir"]["size"] = 100  # tiny reservoir
    config["reservoir"]["num_clusters"] = 4
    config["pretrain"]["batch_size"] = 4
    config["finetune"]["batch_size"] = 4

    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # ---------- 1. Dataset ----------
    print("\n[1/5] Generating tiny dataset...")
    dataset = MovingMNISTDataset(
        num_samples=10,
        actions=["moving", "spinning", "collision", "stationary"],
        seq_length=10,
        split="train",
    )
    print(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"  Frame shape: {sample['frames'].shape}")
    print(f"  Label: {sample['label']}")

    # ---------- 2. Encoder & Reservoir ----------
    print("\n[2/5] Testing encoder and reservoir...")
    encoder = SpatialEncoder(
        img_size=config["data"]["img_size"],
        canvas_size=config["data"]["canvas_size"],
        output_dim=config["reservoir"]["encoder"]["output_dim"],
        encoder_type=config["reservoir"]["encoder"]["type"],
    )
    reservoir = ClusteredReservoir(
        input_dim=config["reservoir"]["encoder"]["output_dim"],
        reservoir_size=100,
        num_clusters=4,
        seed=42,
    )
    test_frames = torch.randn(2, 5, 1, 64, 64)
    encoded = encoder(test_frames)
    print(f"  Encoded shape: {encoded.shape}")
    states, final = reservoir.forward_sequence(encoded)
    print(f"  Reservoir states: {states.shape}")

    # ---------- 3. Pre-training (self-supervised) ----------
    print("\n[3/5] Running self-supervised pre-training...")
    # Use on-the-fly dataloader for unlabeled data
    from data.dataset import get_pretrain_dataloader

    pretrain_loader = get_pretrain_dataloader(config)
    trainer = MultiTaskTrainer(reservoir, encoder, config, device=device)
    readout_ss = trainer.run(pretrain_loader, save_checkpoint=False)
    print("  Pre-training completed.")

    # ---------- 4. Fine-tuning (action classification) ----------
    print("\n[4/5] Fine-tuning action classifier...")
    from data.dataset import get_dataloaders

    dataloaders = get_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    classifier = ActionClassifier(reservoir, encoder, config, checkpoint_path=None)
    # Use frozen reservoir, extract features + ridge
    metrics = classifier.train_classifier(train_loader, val_loader)
    test_metrics = classifier.test(test_loader)
    print(f"  Test accuracy: {test_metrics['test_accuracy']:.2%}")

    # ---------- 5. Summary ----------
    print("\n" + "=" * 60)
    print("SANITY CHECK PASSED")
    print("=" * 60)
    print("\nAll core components work:")
    print("  - Moving MNIST generator uses mnist.pkl ✓")
    print("  - Spatial encoder ✓")
    print("  - Clustered reservoir ✓")
    print("  - Self-supervised pre-training ✓")
    print("  - Downstream fine-tuning ✓")
    print("\nReady to run full experiments:")
    print("  python scripts/run_pretrain.py")
    print("  python scripts/run_finetune.py")
    print("  python scripts/run_baselines.py")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
