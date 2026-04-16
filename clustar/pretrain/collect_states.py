"""
State Collector for CluSTAR Pre-training.
Runs reservoir forward pass on all data and collects hidden states.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional


class ReservoirStateCollector:
    """
    Collects reservoir hidden states from all sequences in dataset.
    Used for offline readout training via ridge regression.
    """

    def __init__(
        self,
        reservoir: torch.nn.Module,
        spatial_encoder: torch.nn.Module,
        device: str = "cpu",
    ):
        self.reservoir = reservoir
        self.spatial_encoder = spatial_encoder
        self.device = device

        # Move to device
        self.reservoir.to(device).eval()
        self.spatial_encoder.to(device).eval()

        # Freeze all parameters
        for param in self.reservoir.parameters():
            param.requires_grad = False
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def collect(
        self, dataloader: DataLoader, save_path: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run reservoir over entire dataset and collect states.

        Args:
            dataloader: yields batches of {'frames': [B, T, C, H, W]}
            save_path: optional path to save collected states

        Returns:
            dict with:
              'states': [num_sequences, T, reservoir_size]
              'frames': [num_sequences, T, C, H, W] (original frames)
        """
        all_states = []
        all_frames = []

        print("Collecting reservoir states...")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            frames = batch["frames"].to(self.device)  # [B, T, C, H, W]

            # Encode frames
            B, T = frames.shape[0], frames.shape[1]
            flat_frames = frames.view(B * T, 1, 64, 64)  # Flatten temporal
            encoded = self.spatial_encoder(flat_frames)  # [B*T, D]
            encoded = encoded.view(B, T, -1)  # [B, T, D]

            # Run reservoir
            states, _ = self.reservoir.forward_sequence(encoded)  # [B, T, N]

            all_states.append(states.cpu())
            all_frames.append(frames.cpu())

        all_states = torch.cat(all_states, dim=0)  # [num_sequences, T, N]
        all_frames = torch.cat(all_frames, dim=0)  # [num_sequences, T, C, H, W]

        result = {"states": all_states, "frames": all_frames}

        if save_path:
            torch.save(result, save_path)
            print(f"Saved states to {save_path}")

        return result

    @torch.no_grad()
    def collect_with_targets(
        self, dataloader: DataLoader, task_factory, flatten_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Collect states and compute all pretext targets in one pass.

        Returns:
            dict with:
              'X': [num_samples, reservoir_size] (flattened states per sequence or per timestep)
              'targets': dict of task-specific targets
        """
        all_states_list = []
        all_frames_list = []

        print("Collecting states and computing targets...")
        for batch in tqdm(dataloader):
            frames = batch["frames"].to(self.device)  # [B, T, C, H, W]

            B, T = frames.shape[0], frames.shape[1]
            flat_frames = frames.view(B * T, 1, 64, 64)
            encoded = self.spatial_encoder(flat_frames).view(B, T, -1)

            states, _ = self.reservoir.forward_sequence(encoded)  # [B, T, N]
            all_states_list.append(states.cpu())
            all_frames_list.append(frames.cpu())

        all_states = torch.cat(all_states_list, dim=0)  # [S, T, N]
        all_frames = torch.cat(all_frames_list, dim=0)  # [S, T, C, H, W]
        S = all_states.shape[0]

        # Compute targets
        targets = task_factory.compute_all_targets(all_frames)

        # Convert states to appropriate format per task
        X_dict = {}

        # For tasks that predict next frame → use state at each timestep except last
        if "frame_prediction" in targets:
            # states: [S, T, N] → use states[:, :-1, :] → [S, T-1, N]
            X_fp = all_states[:, :-1, :].reshape(-1, all_states.shape[-1])
            # Reshape frame targets to [num_pairs, C*H*W]
            C, H, W = all_frames.shape[2], all_frames.shape[3], all_frames.shape[4]
            targets["frame_prediction"] = targets["frame_prediction"].reshape(
                -1, C * H * W
            )
            X_dict["frame_prediction"] = X_fp

        # For classification tasks (temporal_order, speed, segmentation) → use state at each timestep
        for task in ["temporal_order", "speed_regression", "segmentation"]:
            if task in targets:
                X_task = all_states.reshape(-1, all_states.shape[-1])  # [S*T, N]
                X_dict[task] = X_task
                if task == "temporal_order":
                    # Need to flatten pair indices accordingly
                    pair_indices = targets[task][
                        "pair_indices"
                    ]  # [num_pairs_total, 2] with per-seq time indices
                    # Convert global sequence-time indices to flat state index
                    # pair_indices are time indices within each sequence; we need to add sequence offset
                    # We assume pairs are ordered: first all pairs for sequence 0, then seq 1, etc.
                    num_pairs_total = pair_indices.shape[0]
                    pairs_per_seq = num_pairs_total // S  # integer division
                    # Create sequence offset for each pair
                    seq_offsets = torch.arange(
                        S, device=pair_indices.device
                    ).repeat_interleave(pairs_per_seq)
                    flat_idx = (seq_offsets.unsqueeze(1) * T + pair_indices).reshape(
                        -1, 2
                    )
                    targets[task] = {
                        "pair_indices_flat": flat_idx,
                        "labels": targets[task]["labels"],
                    }

        # For rotation contrast → use pairs of states
        if "rotation_contrast" in targets:
            # Map sequence-time indices to flat state index
            S, T = all_states.shape[0], all_states.shape[1]
            time_idx = torch.arange(S).unsqueeze(1) * T + torch.arange(T)
            anchor_idx = time_idx.flatten()[targets["rotation_contrast"]["anchor_idx"]]
            pos_idx = time_idx.flatten()[targets["rotation_contrast"]["pos_idx"]]
            neg_idx = time_idx.flatten()[targets["rotation_contrast"]["neg_idx"]]
            targets["rotation_contrast"] = {
                "anchor_idx": anchor_idx,
                "pos_idx": pos_idx,
                "neg_idx": neg_idx,
            }
            X_dict["rotation_contrast"] = all_states.reshape(-1, all_states.shape[-1])

        return {
            "states": all_states,
            "frames": all_frames,
            "X": X_dict,
            "targets": targets,
        }


def test_collector():
    """Quick test of state collector."""
    from models.reservoir import ClusteredReservoir
    from models.encoder import SpatialEncoder
    import yaml

    with open("configs/reservoir.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize components
    reservoir = ClusteredReservoir(
        input_dim=config["reservoir"]["encoder"]["output_dim"],
        reservoir_size=config["reservoir"]["size"],
        num_clusters=config["reservoir"]["num_clusters"],
        seed=42,
    )
    encoder = SpatialEncoder(
        img_size=config["data"]["img_size"],
        canvas_size=config["data"]["canvas_size"],
        output_dim=config["reservoir"]["encoder"]["output_dim"],
        encoder_type=config["reservoir"]["encoder"]["type"],
    )

    collector = ReservoirStateCollector(reservoir, encoder, device="cpu")

    # Dummy dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {"frames": torch.rand(30, 1, 64, 64)}

    dummy_loader = DataLoader(DummyDataset(), batch_size=4, shuffle=False)
    results = collector.collect(dummy_loader)

    print(f"Collected states: {results['states'].shape}")
    print(f"Collected frames: {results['frames'].shape}")
    assert results["states"].shape[0] == 10

    print("Collector test passed!")


if __name__ == "__main__":
    test_collector()
