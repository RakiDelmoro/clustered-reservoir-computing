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
        actions = ["moving", "spinning", "stationary"]

    data_config = config["data"]
    seq_length = data_config["seq_length"]

    # Create datasets
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

    # Dataloader settings
    # Use finetune config if present, else defaults
    finetune_cfg = config.get("finetune", {})
    batch_size = finetune_cfg.get("batch_size", 64)
    num_workers = finetune_cfg.get("num_workers", 4)

    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if config["training"]["device"] == "cuda" else False,
        )

    return dataloaders
