import sys
import os
import json
import time
import numpy as np
import torch
import imageio
import yaml
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from clustar.models.encoder import SpatiotemporalEncoder
from clustar.models.reservoir import ClusteredReservoir
from clustar.models.deep_reservoir import DeepReservoir
from clustar.models.readout import ActionReadout

ACTIONS = ["moving", "spinning", "stationary"]


def main():
    output_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    video_path = os.path.join(output_dir, "test.mp4")
    meta_path = os.path.join(output_dir, "test_meta.json")
    ckpt_path = os.path.join(output_dir, "checkpoints", "clustar.pt")
    config_path = os.path.join(output_dir, "clustar", "configs", "reservoir.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(meta_path, "r") as f:
        metadata = json.load(f)
    frame_labels = metadata["frame_labels"]

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_config = ckpt["config"]
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)

    enc_cfg = saved_config["reservoir"]["encoder"]
    encoder = (
        SpatiotemporalEncoder(
            canvas_size=saved_config["data"]["canvas_size"],
            spatial_dim=enc_cfg["spatial_dim"],
            temporal_dim=enc_cfg["temporal_dim"],
            seed=saved_config["training"]["seed"],
        )
        .eval()
        .to(device)
    )

    res_cfg = saved_config["reservoir"]
    layer1 = (
        ClusteredReservoir(
            input_dim=enc_cfg["spatial_dim"] + enc_cfg["temporal_dim"],
            reservoir_size=res_cfg["size"],
            num_clusters=res_cfg["num_clusters"],
            cluster_connectivity=res_cfg["cluster"]["within_prob"],
            inter_cluster_connectivity=res_cfg["cluster"]["between_prob"],
            spectral_radius_global=res_cfg["spectral_radius_global"],
            input_scaling=res_cfg["input_scaling"],
            bias_scaling=res_cfg["bias_scaling"],
            seed=saved_config["training"]["seed"] + 10,
            activation=res_cfg.get(
                "activation", config["reservoir"].get("activation", "mixed")
            ),
        )
        .eval()
        .to(device)
    )

    l2_cfg = res_cfg["layer2"]
    layer2 = (
        ClusteredReservoir(
            input_dim=res_cfg["size"] * 2,
            reservoir_size=l2_cfg["size"],
            num_clusters=l2_cfg["num_clusters"],
            cluster_connectivity=l2_cfg["cluster"]["within_prob"],
            inter_cluster_connectivity=l2_cfg["cluster"]["between_prob"],
            spectral_radius_global=l2_cfg["spectral_radius_global"],
            input_scaling=l2_cfg["input_scaling"],
            bias_scaling=l2_cfg["bias_scaling"],
            seed=saved_config["training"]["seed"] + 20,
            activation=l2_cfg.get(
                "activation", config["reservoir"]["layer2"].get("activation", "mixed")
            ),
        )
        .eval()
        .to(device)
    )

    reservoir = DeepReservoir(layer1, layer2)

    l1_size = res_cfg["size"]
    l2_size = l2_cfg["size"]
    feature_dim = l1_size * 4 + l2_size * 4
    num_classes = saved_config["finetune"]["num_classes"]
    readout = ActionReadout(feature_dim=feature_dim, num_classes=num_classes).to(device)
    readout.load_state_dict(ckpt["readout_state_dict"])
    readout.eval()

    cluster_assignments_L1 = reservoir.layer1.cluster_assignments
    num_clusters_L1 = reservoir.layer1.num_clusters
    cluster_assignments_L2 = reservoir.layer2.cluster_assignments
    num_clusters_L2 = reservoir.layer2.num_clusters
    seq_length = saved_config["data"].get("seq_length", 60)

    print(f"\nLoading video: {video_path}")
    video_frames = imageio.mimread(video_path)
    print(f"Video: {len(video_frames)} frames\n")

    state_L1 = torch.zeros(1, reservoir.reservoir_size_L1, device=device)
    state_L2 = torch.zeros(1, reservoir.reservoir_size_L2, device=device)
    seq_length = saved_config["data"].get("seq_length", 60)

    zero_state_L1 = torch.zeros(1, reservoir.reservoir_size_L1, device=device)
    zero_state_L2 = torch.zeros(1, reservoir.reservoir_size_L2, device=device)
    state_window_L1 = deque(maxlen=seq_length)
    state_window_L2 = deque(maxlen=seq_length)
    for _ in range(seq_length - 1):
        state_window_L1.append(zero_state_L1.clone())
        state_window_L2.append(zero_state_L2.clone())
    prev_frame_tensor = None

    MIN_WARMUP = 10

    print("=" * 68)
    print("  CluSTAR Deep Reservoir — Autoregressive Inference")
    print("=" * 68)
    print(f"  {'Frame':>5}  {'Target':<18}  {'Predicted':<14}  Confidence")
    print("-" * 68)

    fps = 10.0
    frame_delay = 1.0 / fps
    correct = 0
    total_predicted = 0
    per_class_correct = {a: 0 for a in ACTIONS}
    per_class_total = {a: 0 for a in ACTIONS}

    for t, frame in enumerate(video_frames):
        t_start = time.time()

        frame_np = np.array(frame)
        if frame_np.ndim == 3:
            frame_np = frame_np[:, :, 0]
        frame_tensor = (
            torch.from_numpy(frame_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        )
        frame_tensor = frame_tensor.to(device)

        with torch.no_grad():
            if prev_frame_tensor is None:
                frame_pair = frame_tensor.unsqueeze(1)
            else:
                frame_pair = torch.stack([prev_frame_tensor, frame_tensor], dim=1)
            encoded = encoder(frame_pair)
            encoded_t = encoded[:, -1, :]
            prev_frame_tensor = frame_tensor.clone()

            state_L1, state_L2 = reservoir(encoded_t, state_L1, state_L2)
            state_window_L1.append(state_L1.clone())
            state_window_L2.append(state_L2.clone())

        target = frame_labels[t]

        if t < MIN_WARMUP:
            print(f"  {t + 1:>5}  {target:<18}  {'(warmup)':<14}  ---")
            elapsed = time.time() - t_start
            time.sleep(max(0, frame_delay - elapsed))
            continue

        with torch.no_grad():
            states_L1_stack = torch.cat(list(state_window_L1), dim=0).unsqueeze(0)
            states_L2_stack = torch.cat(list(state_window_L2), dim=0).unsqueeze(0)

            cluster_feats = []
            for c in range(num_clusters_L1):
                mask = cluster_assignments_L1 == c
                c_states = states_L1_stack[:, :, mask]
                c0 = c_states[:, 0, :]
                cT = c_states[:, -1, :]
                c_mean = c_states.mean(dim=1)
                c_max = c_states.max(dim=1)[0]
                cluster_feats.extend([c0, cT, c_mean, c_max])

            for c in range(num_clusters_L2):
                mask = cluster_assignments_L2 == c
                c_states = states_L2_stack[:, :, mask]
                c0 = c_states[:, 0, :]
                cT = c_states[:, -1, :]
                c_mean = c_states.mean(dim=1)
                c_max = c_states.max(dim=1)[0]
                cluster_feats.extend([c0, cT, c_mean, c_max])

            feats = torch.cat(cluster_feats, dim=1)

            feats_scaled = (feats - feat_mean) / feat_std
            logits = readout(feats_scaled)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        action = ACTIONS[pred_idx]
        match = "✓" if action == target else "✗"
        if action == target:
            correct += 1
        total_predicted += 1
        per_class_total[target] = per_class_total.get(target, 0) + 1
        if action == target:
            per_class_correct[target] = per_class_correct.get(target, 0) + 1
        bar = "█" * int(confidence * 20)
        print(
            f"  {t + 1:>5}  {target:<18}  {action:<14}  {confidence:>6.1%} {bar} {match}"
        )

        elapsed = time.time() - t_start
        sleep_time = max(0, frame_delay - elapsed)
        time.sleep(sleep_time)

    print("-" * 68)
    if total_predicted > 0:
        accuracy = correct / total_predicted
        print(
            f"  Inference complete. Accuracy: {correct}/{total_predicted} ({accuracy:.1%})"
        )
        print()
        print("  Per-class recall:")
        for a in ACTIONS:
            if per_class_total.get(a, 0) > 0:
                recall = per_class_correct.get(a, 0) / per_class_total[a]
                print(
                    f"    {a:>12s}: {per_class_correct.get(a, 0)}/{per_class_total[a]} ({recall:.1%})"
                )
    else:
        print("  No predictions made.")


if __name__ == "__main__":
    main()
