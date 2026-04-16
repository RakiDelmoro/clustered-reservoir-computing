"""
Self-Supervised Pretext Tasks for CluSTAR Pre-training.
Each task defines: (1) target computation from frames, (2) loss function.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.decomposition import PCA


class PretextTaskFactory:
    """
    Factory for creating pretext task data loaders.
    Each task takes reservoir states and produces training targets.
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.tasks_config = config["pretrain"]["tasks"]

        # For shape features (Hu moments or PCA)
        self.shape_encoder = None
        if config["data"].get("use_shape_pca", False):
            self._fit_shape_pca(config)

    def _fit_shape_pca(self, config):
        """Fit PCA on digit shapes for compact encoding."""
        # Placeholder: would fit on MNIST digits
        pass

    def compute_frame_prediction_targets(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Task: Predict next frame I_{t+1} from states at time t.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            targets: [B*(T-1), H*W] flattened next frames
        """
        # Predict frame t+1 from state at t
        # Use all but last frame as input, predict all but first frame
        B, T = frames.shape[0], frames.shape[1]
        targets = frames[:, 1:, :, :, :]  # [B, T-1, C, H, W]
        targets = targets.reshape(
            -1, frames.shape[2] * frames.shape[3] * frames.shape[4]
        )
        return targets  # [B*(T-1), 784]

    def compute_temporal_order_targets(
        self, frames: torch.Tensor, delta_range: Tuple[int, int] = (2, 5)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Task: Given two windows, which came first in time?

        Creates pairs: (I_t, I_{t+Δ}) -> label=1 (correct order)
                      (I_{t+Δ}, I_t) -> label=0 (shuffled)

        Args:
            frames: [B, T, C, H, W]
            delta_range: range of frame offsets Δ

        Returns:
            state_pairs: [num_pairs, 2, reservoir_size] (will be filled by caller)
            labels: [num_pairs] binary
        """
        B, T = frames.shape[0], frames.shape[1]
        pairs = []
        labels = []

        for b in range(B):
            for t in range(T - max(delta_range)):
                delta = np.random.randint(delta_range[0], delta_range[1] + 1)
                if t + delta >= T:
                    continue

                # Pair 1: correct order
                pairs.append((t, t + delta))
                labels.append(1)

                # Pair 2: shuffled order
                pairs.append((t + delta, t))
                labels.append(0)

        # Convert to tensors (indices; caller will gather states)
        indices = torch.tensor(pairs, dtype=torch.long)  # [num_pairs, 2]
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        return indices, labels_tensor

    def compute_speed_targets(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Task: Predict speed (magnitude of motion) at each frame.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            speeds: [B*T] scalar speeds
        """
        B, T = frames.shape[0], frames.shape[1]
        speeds = []

        for b in range(B):
            for t in range(T):
                if t == 0:
                    speed = 0.0
                else:
                    # Frame difference
                    prev = frames[b, t - 1, 0]
                    curr = frames[b, t, 0]
                    # Compute optical flow (simplified: centroid displacement)
                    prev_mask = (prev > 0.5).float()
                    curr_mask = (curr > 0.5).float()

                    # Get centroids
                    y_coords = torch.arange(frames.shape[3]).float()
                    x_coords = torch.arange(frames.shape[4]).float()

                    if prev_mask.sum() > 0 and curr_mask.sum() > 0:
                        y_prev = (prev_mask * y_coords[:, None]).sum() / prev_mask.sum()
                        x_prev = (prev_mask * x_coords[None, :]).sum() / prev_mask.sum()
                        y_curr = (curr_mask * y_coords[:, None]).sum() / curr_mask.sum()
                        x_curr = (curr_mask * x_coords[None, :]).sum() / curr_mask.sum()
                        speed = torch.sqrt(
                            (x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2
                        ).item()
                    else:
                        speed = 0.0
                speeds.append(speed)

        return torch.tensor(speeds, dtype=torch.float32).unsqueeze(1)  # [B*T, 1]

    def compute_segmentation_targets(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Task: Segment digit from background (binary mask).

        Args:
            frames: [B, T, C, H, W]

        Returns:
            masks: [B*T, H*W] binary (0=background, 1=digit)
        """
        masks = (frames > 0.5).float()
        masks = masks.reshape(-1, frames.shape[2] * frames.shape[3] * frames.shape[4])
        return masks

    def compute_rotation_contrast_pairs(
        self, frames: torch.Tensor, metadata: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Task: Contrastive learning — same digit under small rotation should be similar.
        Simplified: use frame similarity as proxy (not true rotation augmentation).

        Args:
            frames: [B, T, C, H, W]
            metadata: list of dicts with rotation info

        Returns:
            anchor_idx, pos_idx, neg_idx: indices into flattened (B*T) dimension
        """
        # Simplified: anchor = frame t, positive = frame t+1 (small motion = small rotation)
        # negative = frame t+k (k large, different orientation)
        B, T = frames.shape[0], frames.shape[1]
        anchors = []
        positives = []
        negatives = []

        for b in range(B):
            for t in range(T - 3):
                anchors.append(b * T + t)
                positives.append(b * T + t + 1)  # Next frame (small rotation)
                # Negative: random far frame
                neg_t = np.random.randint(t + 3, min(T, t + 10))
                negatives.append(b * T + neg_t)

        anchor_idx = torch.tensor(anchors, dtype=torch.long)
        pos_idx = torch.tensor(positives, dtype=torch.long)
        neg_idx = torch.tensor(negatives, dtype=torch.long)

        return anchor_idx, pos_idx, neg_idx

    def compute_all_targets(
        self, frames: torch.Tensor, metadata: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all pretext task targets for a batch of frames.

        Args:
            frames: [B, T, C, H, W]
            metadata: optional list of metadata dicts

        Returns:
            targets: dict mapping task_name -> target tensor
        """
        targets = {}

        if self.tasks_config.get("frame_prediction", {}).get("enabled", False):
            targets["frame_prediction"] = self.compute_frame_prediction_targets(frames)

        if self.tasks_config.get("temporal_order", {}).get("enabled", False):
            indices, labels = self.compute_temporal_order_targets(frames)
            targets["temporal_order"] = {
                "pair_indices": indices,  # [num_pairs, 2]
                "labels": labels,  # [num_pairs]
            }

        if self.tasks_config.get("speed_regression", {}).get("enabled", False):
            targets["speed_regression"] = self.compute_speed_targets(frames)

        if self.tasks_config.get("segmentation", {}).get("enabled", False):
            targets["segmentation"] = self.compute_segmentation_targets(frames)

        if self.tasks_config.get("rotation_contrast", {}).get("enabled", False):
            a_idx, p_idx, n_idx = self.compute_rotation_contrast_pairs(frames, metadata)
            targets["rotation_contrast"] = {
                "anchor_idx": a_idx,
                "pos_idx": p_idx,
                "neg_idx": n_idx,
            }

        return targets


def test_tasks():
    """Test pretext task computation."""
    config = {
        "pretrain": {
            "tasks": {
                "frame_prediction": {"enabled": True},
                "temporal_order": {"enabled": True, "delta_range": [2, 4]},
                "speed_regression": {"enabled": True},
                "segmentation": {"enabled": True},
                "rotation_contrast": {"enabled": True},
            }
        }
    }

    factory = PretextTaskFactory(config)

    # Dummy data
    B, T = 4, 30
    frames = torch.rand(B, T, 1, 64, 64)
    frames = (frames > 0.8).float()  # Binarize to simulate digit

    targets = factory.compute_all_targets(frames)

    print("Pretext targets:")
    for task, tgt in targets.items():
        if isinstance(tgt, dict):
            print(f"  {task}: keys={list(tgt.keys())}")
            for k, v in tgt.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}")
        elif isinstance(tgt, torch.Tensor):
            print(f"  {task}: shape={tgt.shape}")

    print("Task tests passed!")


if __name__ == "__main__":
    test_tasks()
