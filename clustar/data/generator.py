"""
Moving MNIST Dataset Generator
Synthesizes sequences of moving MNIST digits with 4 action types.
Uses pre-loaded mnist.pkl for digit templates.
"""

import torch
import numpy as np
import pickle
import os
from typing import Tuple, List, Dict, Optional
import random


class MovingMNISTGenerator:
    """
    Generates synthetic sequences of moving MNIST digits.
    Actions: moving, spinning, collision, stationary
    """

    def __init__(
        self,
        img_size: int = 28,
        canvas_size: int = 64,
        num_digits: int = 1,
        digit_class_range: Tuple[int, int] = (0, 9),
        seed: int = 42,
        mnist_pkl_path: Optional[str] = None,
    ):
        self.img_size = img_size
        self.canvas_size = canvas_size
        self.num_digits = num_digits
        self.digit_class_range = digit_class_range
        self.rng = np.random.RandomState(seed)

        # Determine default path: project root / mnist.pkl
        if mnist_pkl_path is None:
            # Find project root relative to this file (data/generator.py -> project root)
            this_file = os.path.abspath(__file__)
            # this_file = /workspaces/foundation-model-v1/clustar/data/generator.py
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
            mnist_pkl_path = os.path.join(project_root, "mnist.pkl")
        self.mnist_pkl_path = mnist_pkl_path

        # Load MNIST digit templates from pickle (no download)
        self.digit_images = self._load_mnist_templates(mnist_pkl_path)

    def _load_mnist_templates(self, pkl_path: str) -> Dict[int, torch.Tensor]:
        """
        Load digit templates from mnist.pkl.
        Expected format: pickle containing (train, val, test) splits.
        Each split is (images: [N,784], labels: [N]).
        Returns one clean template per digit class (first occurrence).
        """
        # Resolve path relative to project root if needed
        pkl_path = os.path.abspath(pkl_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # data[0] = train split: (images, labels)
        train_images, train_labels = data[0]
        train_images = torch.from_numpy(train_images).float()
        train_labels = torch.from_numpy(train_labels).long()

        # Reshape to 28x28 and binarize
        templates = {}
        for label in range(10):
            idxs = (train_labels == label).nonzero(as_tuple=True)[0]
            if len(idxs) > 0:
                idx = idxs[0].item()  # first occurrence
                img = train_images[idx].view(1, 28, 28)  # add channel dim (1,28,28)
                templates[label] = (img > 0.5).float()

        return templates

    def generate_sequence(
        self, action: str, seq_length: int = 30, fps: float = 10.0
    ) -> Dict:
        """
        Generate one sequence with specified action.

        Returns:
            frames: [seq_length, 1, canvas_size, canvas_size]
            metadata: dict with action, velocities, positions, etc.
        """
        assert action in ["moving", "spinning", "collision", "stationary"]

        # Initialize canvas
        frames = torch.zeros(seq_length, 1, self.canvas_size, self.canvas_size)

        # Initialize digit properties
        if action == "collision" and self.num_digits >= 2:
            num_digits_used = 2
        else:
            num_digits_used = self.num_digits

        digits = []
        for d in range(num_digits_used):
            digit_class = self.rng.randint(*self.digit_class_range)
            digit_img = self.digit_images[digit_class]

            # Initial position (top-left corner of digit on canvas)
            pos_x = self.rng.randint(0, self.canvas_size - self.img_size)
            pos_y = self.rng.randint(0, self.canvas_size - self.img_size)

            # Velocity
            v_config = {
                "moving": (2, 6),
                "spinning": (0, 1),
                "collision": (3, 7),
                "stationary": (0, 0.5),
            }[action]
            speed = self.rng.uniform(*v_config)
            angle = self.rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            # Rotation
            rot_config = {
                "moving": False,
                "spinning": True,
                "collision": False,
                "stationary": False,
            }[action]
            if rot_config:
                omega = self.rng.uniform(5, 20) * (1 if self.rng.rand() > 0.5 else -1)
            else:
                omega = 0.0

            digits.append(
                {
                    "class": digit_class,
                    "image": digit_img,
                    "pos": np.array([pos_x, pos_y], dtype=float),
                    "vel": np.array([vx, vy], dtype=float),
                    "angle": 0.0,
                    "omega": omega,
                }
            )

        # Simulate motion
        for t in range(seq_length):
            canvas = torch.zeros(1, self.canvas_size, self.canvas_size)

            for d_info in digits:
                # Update position
                d_info["pos"] += d_info["vel"] * (1.0 / fps)

                # Update angle
                d_info["angle"] += d_info["omega"] * (1.0 / fps)
                d_info["angle"] %= 360

                # Bounce off walls
                for dim in [0, 1]:  # x, y
                    if d_info["pos"][dim] < 0:
                        d_info["pos"][dim] = 0
                        d_info["vel"][dim] *= -1
                    elif d_info["pos"][dim] > self.canvas_size - self.img_size:
                        d_info["pos"][dim] = self.canvas_size - self.img_size
                        d_info["vel"][dim] *= -1

                # Render digit onto canvas (simplified: no rotation for speed)
                # For rotation, you'd need scipy.ndimage.rotate
                x, y = int(d_info["pos"][0]), int(d_info["pos"][1])
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(self.canvas_size, x + self.img_size)
                y2 = min(self.canvas_size, y + self.img_size)
                dx1 = max(0, -x)
                dy1 = max(0, -y)
                dx2 = self.img_size - max(0, (x + self.img_size) - self.canvas_size)
                dy2 = self.img_size - max(0, (y + self.img_size) - self.canvas_size)

                if x2 > x1 and y2 > y1:
                    canvas[0, y1:y2, x1:x2] = torch.maximum(
                        canvas[0, y1:y2, x1:x2], d_info["image"][0, dy1:dy2, dx1:dx2]
                    )

            frames[t] = canvas

        # Check collision (if 2 digits)
        collision_flag = False
        if num_digits_used == 2:
            d0, d1 = digits[0], digits[1]
            pos0, size0 = d0["pos"], self.img_size
            pos1, size1 = d1["pos"], self.img_size
            # Overlap check
            if abs(pos0[0] - pos1[0]) < size0 and abs(pos0[1] - pos1[1]) < size0:
                collision_flag = True

        # Compile metadata
        avg_speed = np.mean([np.linalg.norm(d["vel"]) for d in digits])
        has_rotation = any(d["omega"] != 0 for d in digits)

        metadata = {
            "action": action,
            "action_label": ["moving", "spinning", "collision", "stationary"].index(
                action
            ),
            "num_digits": num_digits_used,
            "avg_speed": float(avg_speed),
            "has_rotation": has_rotation,
            "collision": collision_flag if action == "collision" else False,
            "digit_classes": [d["class"] for d in digits],
            "trajectories": [d["pos"].copy() for d in digits],
        }

        return {"frames": frames, "metadata": metadata}

    def generate_batch(
        self, actions: List[str], batch_size: int, seq_length: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """Generate a batch of sequences."""
        frames_list = []
        labels_list = []
        metas = []

        for _ in range(batch_size):
            action = self.rng.choice(actions)
            sample = self.generate_sequence(action, seq_length)
            frames_list.append(sample["frames"])
            labels_list.append(sample["metadata"]["action_label"])
            metas.append(sample["metadata"])

        frames_batch = torch.stack(frames_list)  # [B, T, 1, H, W]
        labels_batch = torch.tensor(labels_list, dtype=torch.long)  # [B]

        return frames_batch, labels_batch, metas


# PyTorch Dataset wrapper (on-the-fly generation)
class MovingMNISTDataset(torch.utils.data.Dataset):
    """
    On-the-fly moving MNIST dataset.
    Generates each sequence deterministically based on index (no caching).
    """

    def __init__(
        self,
        num_samples: int,
        actions: List[str],
        seq_length: int = 30,
        split: str = "train",
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.actions = actions
        self.seq_length = seq_length
        self.split = split

        # Split-specific seed offset
        split_offset = {"train": 0, "val": 1000, "test": 2000}.get(split, 0)
        self.base_seed = seed + split_offset

        # Create generator (digit templates loaded from pickle once)
        self.generator = MovingMNISTGenerator(
            img_size=28,
            canvas_size=64,
            num_digits=1,
            digit_class_range=(0, 9),
            seed=self.base_seed,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a deterministic sequence for given index.
        Uses per-index RNG seeded by base_seed + idx.
        """
        # Per-index RNG (ensures reproducibility regardless of sampling order)
        rng = np.random.RandomState(self.base_seed + idx)
        original_rng = self.generator.rng
        self.generator.rng = rng
        try:
            action = rng.choice(self.actions)
            sample = self.generator.generate_sequence(action, self.seq_length)
        finally:
            self.generator.rng = original_rng

        return {
            "frames": sample["frames"],  # [T, 1, H, W]
            "label": sample["metadata"]["action_label"],
            "metadata": sample["metadata"],
        }


def main_test():
    """Test dataset generation."""
    generator = MovingMNISTGenerator(seed=42)
    sample = generator.generate_sequence("moving", seq_length=30)
    print("Frames shape:", sample["frames"].shape)
    print("Metadata:", sample["metadata"])

    # Generate dataset
    dataset = MovingMNISTDataset(
        num_samples=100,
        actions=["moving", "spinning", "collision", "stationary"],
        seq_length=30,
        split="train",
    )
    print("Dataset size:", len(dataset))
    batch = dataset[0]
    print("Batch frames:", batch["frames"].shape)
    print("Batch label:", batch["label"])


if __name__ == "__main__":
    main_test()
