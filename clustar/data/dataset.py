"""
PyTorch Dataset and DataLoader for Moving MNIST.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from .generator import MovingMNISTDataset
from typing import List, Dict
import yaml


def get_dataloaders(config: Dict, actions: List[str] = None) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        config: Configuration dict (from YAML)
        actions: List of action labels (default: all 4)

    Returns:
        dict with 'train', 'val', 'test' dataloaders
    """
    if actions is None:
        actions = ["moving", "spinning", "collision", "stationary"]

    data_config = config["data"]
    seq_length = data_config["seq_length"]

    # Create datasets (cached on disk for efficiency)
    datasets = {}
    for split, num_samples in [
        ("train", data_config["train_samples"]),
        ("val", data_config["val_samples"]),
        ("test", data_config["test_samples"]),
    ]:
        datasets[split] = MovingMNISTDataset(
            num_samples=num_samples,
            actions=actions,
            seq_length=seq_length,
            split=split,
            seed=config["training"]["seed"],
        )

    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config["pretrain"]["batch_size"],
            shuffle=shuffle,
            num_workers=config["pretrain"]["num_workers"],
            pin_memory=True if config["training"]["device"] == "cuda" else False,
        )

    return dataloaders


def get_pretrain_dataloader(config: Dict) -> DataLoader:
    """
    Dataloader for self-supervised pre-training (unlabeled data).
    Generates separate unlabeled sequences (no action labels needed).
    """

    class PretrainDataset(torch.utils.data.Dataset):
        """
        On-the-fly unlabeled sequence generator for pre-training.
        No caching to keep memory footprint low.
        """

        def __init__(self, num_samples: int, seq_length: int, seed: int):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.base_seed = seed
            # Generator will be created per-index in __getitem__ using a lightweight approach
            # We store the base seed and create generator per sample

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Deterministic RNG per index
            rng = np.random.RandomState(self.base_seed + idx)
            # Create a fresh generator for this sample (digit templates are cached globally)
            # We can reuse a singleton generator but need thread safety? Simpler: create new each time
            # To avoid re-loading templates each time, we rely on generator caching at import level?
            # Instead: instantiate a generator once and clone its digit_images?
            # We'll create a generator lazily and cache its digit_images
            if not hasattr(self, "_cached_generator"):
                self._cached_generator = MovingMNISTGenerator(seed=self.base_seed)
            gen = self._cached_generator
            # Temporarily replace rng
            original_rng = gen.rng
            gen.rng = rng
            try:
                action = rng.choice(["moving", "spinning", "collision", "stationary"])
                sample = gen.generate_sequence(action, self.seq_length)
            finally:
                gen.rng = original_rng
            return {"frames": sample["frames"]}

    dataset = PretrainDataset(
        num_samples=config["data"]["pretrain_samples"],
        seq_length=config["data"]["seq_length"],
        seed=config["training"]["seed"] + 3000,
    )

    return DataLoader(
        dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        num_workers=config["pretrain"]["num_workers"],
        pin_memory=True if config["training"]["device"] == "cuda" else False,
    )


# Import here to avoid circular dependency
from .generator import MovingMNISTGenerator


if __name__ == "__main__":
    # Test dataloaders
    with open("configs/reservoir.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataloaders = get_dataloaders(config)
    for split, loader in dataloaders.items():
        batch = next(iter(loader))
        print(f"{split}: frames {batch['frames'].shape}, labels {batch['label'].shape}")

    pretrain_loader = get_pretrain_dataloader(config)
    batch = next(iter(pretrain_loader))
    print(f"pretrain: frames {batch['frames'].shape}")
