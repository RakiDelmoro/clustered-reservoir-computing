#!/usr/bin/env python3
"""
Diagnostic: Trace where class signal is destroyed in the pipeline.
Tests each stage: encoder → reservoir states → aggregated features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from clustar.data.generator import MovingMNISTGenerator
from clustar.models.encoder import SpatiotemporalEncoder
from clustar.models.reservoir import ClusteredReservoir

ACTIONS = ["moving", "spinning", "stationary"]


def test_stage(X, y, label):
    """Quick linear probe at a given pipeline stage."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, normalize="true")
    print(f"\n  {label}")
    print(f"    Accuracy: {acc:.2%}  |  Feature dim: {X.shape[1]}")
    print(f"    Confusion matrix (recall):")
    for i, action in enumerate(ACTIONS):
        row = " ".join(f"{cm[i, j]:.2f}" for j in range(3))
        print(f"      {action:>12s} | {row}")
    return acc


def main():
    print("=" * 60)
    print("PIPELINE SIGNAL PRESERVATION DIAGNOSTIC")
    print("=" * 60)

    # Config
    encoder_dim = 768
    spatial_dim = 512
    temporal_dim = 256
    reservoir_size = 1000
    num_clusters = 10
    num_per_action = 100
    seq_length = 30

    # Build components
    encoder = SpatiotemporalEncoder(
        canvas_size=64, spatial_dim=spatial_dim, temporal_dim=temporal_dim, seed=42
    )
    reservoir = ClusteredReservoir(
        input_dim=spatial_dim + temporal_dim,
        reservoir_size=reservoir_size,
        num_clusters=num_clusters,
        activation="mixed",
        seed=42,
    )

    # Generate data
    gen = MovingMNISTGenerator(seed=99, num_digits=1)

    encoded_all = []  # [B, T, D] mean over time
    states_last = []  # reservoir last state
    states_mean = []  # reservoir mean state
    states_concat = []  # [first, last, mean, max]
    states_per_cluster = []  # per-cluster stats
    labels = []

    for action in ACTIONS:
        for _ in range(num_per_action):
            sample = gen.generate_sequence(action, seq_length=seq_length)
            frames = sample["frames"]  # [T, 1, H, W]
            label = ACTIONS.index(action)

            with torch.no_grad():
                encoded = encoder(frames.unsqueeze(0))  # [1, T, D]
                encoded = encoded.squeeze(0)  # [T, D]
                states, _ = reservoir.forward_sequence(
                    encoded.unsqueeze(0)
                )  # [1, T, N]
                states = states.squeeze(0)  # [T, N]

            # Stage 1: encoder output (mean over time)
            encoded_all.append(encoded.mean(dim=0).numpy())

            # Stage 2: reservoir last state
            states_last.append(states[-1].numpy())

            # Stage 3: reservoir mean state
            states_mean.append(states.mean(dim=0).numpy())

            # Stage 4: concat aggregation
            x0 = states[0].numpy()
            xT = states[-1].numpy()
            x_mean = states.mean(dim=0).numpy()
            x_max = states.max(dim=0).values.numpy()
            states_concat.append(np.concatenate([x0, xT, x_mean, x_max]))

            # Stage 5: per-cluster aggregation
            cluster_assignments = reservoir.cluster_assignments.numpy()
            num_clusters_r = reservoir.num_clusters
            cluster_feats = []
            for c in range(num_clusters_r):
                mask = cluster_assignments == c
                c_states = states[:, mask]
                c0 = c_states[0].numpy()
                cT = c_states[-1].numpy()
                c_mean = c_states.mean(dim=0).numpy()
                c_max = c_states.max(dim=0).values.numpy()
                cluster_feats.extend([c0, cT, c_mean, c_max])
            states_per_cluster.append(np.concatenate(cluster_feats))

            labels.append(label)

    y = np.array(labels)

    print(f"\nDataset: {num_per_action} samples × 3 actions = {len(y)} total")
    print(f"Encoder dim: {encoder_dim}, Reservoir size: {reservoir_size}")

    # Test each stage
    test_stage(np.array(encoded_all), y, "Stage 1: Encoder features (mean over time)")
    test_stage(np.array(states_last), y, "Stage 2: Reservoir last state")
    test_stage(np.array(states_mean), y, "Stage 3: Reservoir mean state")
    test_stage(
        np.array(states_concat), y, "Stage 4: Concat aggregation [first,last,mean,max]"
    )
    test_stage(np.array(states_per_cluster), y, "Stage 5: Per-cluster aggregation")

    # Also test raw frame diffs (to see if temporal info helps)
    frame_diffs = []
    for action in ACTIONS:
        for _ in range(num_per_action):
            sample = gen.generate_sequence(action, seq_length=seq_length)
            frames = sample["frames"]
            with torch.no_grad():
                encoded = encoder(frames.unsqueeze(0)).squeeze(0)  # [T, D]
                diffs = encoded[1:] - encoded[:-1]
                feat = torch.cat(
                    [
                        diffs.mean(dim=0),
                        diffs.abs().mean(dim=0),
                    ]
                ).numpy()
            frame_diffs.append(feat)

    test_stage(
        np.array(frame_diffs), y, "Stage 6: Encoder temporal diffs (mean + abs_mean)"
    )


if __name__ == "__main__":
    main()
