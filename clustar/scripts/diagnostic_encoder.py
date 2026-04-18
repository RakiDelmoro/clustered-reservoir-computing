#!/usr/bin/env python3
"""
Diagnostic: Measure how well the encoder preserves action-relevant information.
Tests both 128D and 512D encoders for comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from clustar.data.generator import MovingMNISTGenerator, MovingMNISTDataset
from clustar.models.encoder import SpatiotemporalEncoder

ACTIONS = ["moving", "spinning", "stationary"]


def extract_encoder_features(encoder, dataset, num_samples=200):
    """Extract per-sequence encoder features + labels for a dataset."""
    features = []
    labels = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        frames = sample["frames"]  # [T, 1, H, W]
        label = sample["label"]
        with torch.no_grad():
            encoded = encoder(frames.unsqueeze(0))  # [1, T, D]
        feat = encoded.squeeze(0).mean(dim=0).numpy()
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)


def measure_separability(X, y, label=""):
    """Measure linear separability of features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, normalize="true")

    print(f"\n{label}")
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Confusion matrix (normalized recall):")
    for i, action in enumerate(ACTIONS):
        row = " ".join(f"{cm[i, j]:.2f}" for j in range(3))
        print(f"    {action:>12s} | {row}")

    # Per-class feature variance
    print(f"  Per-class feature norm (mean):")
    for i, action in enumerate(ACTIONS):
        mask = y == i
        norm = np.linalg.norm(X_scaled[mask], axis=1).mean()
        print(f"    {action:>12s}: {norm:.3f}")

    return acc


def measure_temporal_separability(encoder, num_per_action=50):
    """Measure if consecutive frame diffs are distinguishable across actions."""
    gen = MovingMNISTGenerator(seed=99, num_digits=1)

    diffs = []
    labels = []

    for action in ACTIONS:
        for _ in range(num_per_action):
            sample = gen.generate_sequence(action, seq_length=30)
            frames = sample["frames"]  # [T, 1, H, W]
            with torch.no_grad():
                encoded = encoder(frames.unsqueeze(0))  # [1, T, D]
            encoded = encoded.squeeze(0)  # [T, D]
            frame_diffs = encoded[1:] - encoded[:-1]
            feat = torch.cat(
                [
                    frame_diffs.mean(dim=0),
                    frame_diffs.std(dim=0),
                    frame_diffs.abs().mean(dim=0),
                ]
            ).numpy()
            diffs.append(feat)
            labels.append(ACTIONS.index(action))

    return measure_separability(
        np.array(diffs),
        np.array(labels),
        label="Temporal diffs (mean+std+abs_mean of frame diffs)",
    )


def main():
    print("=" * 60)
    print("ENCODER SEPARABILITY DIAGNOSTIC")
    print("=" * 60)

    # Create dataset
    dataset = MovingMNISTDataset(
        num_samples=200,
        actions=ACTIONS,
        seq_length=30,
        split="test",
        seed=2000,
    )

    # Test both encoder sizes
    for spatial, temporal, label in [
        (256, 128, "384D spatiotemporal encoder (256+128)"),
        (512, 256, "768D spatiotemporal encoder (512+256)"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Testing: {label}")
        print(f"{'=' * 60}")

        encoder = SpatiotemporalEncoder(
            canvas_size=64,
            spatial_dim=spatial,
            temporal_dim=temporal,
            seed=42,
        )

        # Static features (mean over time)
        X, y = extract_encoder_features(encoder, dataset, num_samples=200)
        acc_static = measure_separability(X, y, label=f"Static features ({label})")

        # Temporal features
        acc_temporal = measure_temporal_separability(encoder)

        print(f"\n  Summary for {label}:")
        print(f"    Static accuracy:  {acc_static:.2%}")
        print(f"    Temporal accuracy: {acc_temporal:.2%}")

    # Data health check
    print(f"\n{'=' * 60}")
    print("DATA HEALTH CHECK")
    print(f"{'=' * 60}")
    gen = MovingMNISTGenerator(seed=99, num_digits=1)
    for action in ACTIONS:
        samples = [gen.generate_sequence(action, seq_length=30) for _ in range(5)]
        speeds = [s["metadata"]["avg_speed"] for s in samples]
        rotations = [s["metadata"]["has_rotation"] for s in samples]
        n_digits = [s["metadata"]["num_digits"] for s in samples]
        print(
            f"  {action:>12s}: speed={np.mean(speeds):.2f}, "
            f"rotation={sum(rotations)}/5, digits={set(n_digits)}"
        )


if __name__ == "__main__":
    main()
