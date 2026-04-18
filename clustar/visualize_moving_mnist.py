#!/usr/bin/env python3
"""
Generate animated visualization of Moving MNIST sequences.

Creates an animated GIF showing 5 different sequences playing side-by-side.
Each row shows one complete sequence (30 frames) with action label and digit classes.
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from data.generator import MovingMNISTGenerator


def create_animated_gif(
    output_path: str = "visualizations/moving-mnist-example.gif",
    num_sequences: int = 5,
    seq_length: int = 30,
    fps: int = 10,
    upscale_factor: int = 4,
    seed: int = None,
):
    """
    Generate an animated GIF of Moving MNIST sequences.

    Args:
        output_path: Path to save the GIF
        num_sequences: Number of sequences (rows) to show
        seq_length: Number of frames per sequence
        fps: Frames per second (higher = smoother, default 10)
        upscale_factor: Factor to upscale each frame for better visibility
        seed: Random seed for reproducibility (None = random each run)
    """
    frame_duration_ms = int(1000 / fps)  # Convert fps to ms per frame

    # Setup - use random seed if not provided
    if seed is None:
        seed = np.random.randint(0, 2**31)
        print(f"[INFO] Using random seed: {seed}")
    else:
        print(f"[INFO] Using fixed seed: {seed}")
    rng = np.random.RandomState(seed)
    generator = MovingMNISTGenerator(seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define actions - want one of each, plus one extra to reach 5
    all_actions = ["moving", "spinning", "stationary"]
    selected_actions = all_actions[:3]  # First 3 distinct
    if num_sequences > 3:
        selected_actions.append(rng.choice(all_actions))  # 4th as random from set

    # Generate sequences
    sequences = []
    for i in range(num_sequences):
        action = (
            selected_actions[i]
            if i < len(selected_actions)
            else rng.choice(all_actions)
        )
        seq_data = generator.generate_sequence(action, seq_length=seq_length)
        sequences.append(
            {
                "frames": seq_data["frames"],  # [T, 1, H, W]
                "action": action,
                "digit_classes": seq_data["metadata"]["digit_classes"],
            }
        )

    # Prepare frames for GIF
    # Each GIF frame will be a composite: all sequences at the same timestep t, stacked vertically
    single_frame_h = 64 * upscale_factor  # 256
    single_frame_w = 64 * upscale_factor  # 256
    total_height = num_sequences * single_frame_h
    total_width = single_frame_w

    gif_frames = []

    # For each timestep t, create a composite frame showing all sequences at that timestep
    for t in range(seq_length):
        # Create blank canvas
        composite = Image.new(
            "L", (total_width, total_height), color=0
        )  # 'L' = 8-bit grayscale

        for row, seq in enumerate(sequences):
            # Get frame at time t
            frame_tensor = seq["frames"][t]  # [1, 64, 64]
            frame_np = frame_tensor.squeeze().numpy()  # [64, 64], values 0 or 1

            # Upscale using nearest neighbor (preserves binary nature)
            frame_upscaled = np.kron(
                frame_np, np.ones((upscale_factor, upscale_factor))
            )
            frame_uint8 = (frame_upscaled * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame_uint8, mode="L")

            # Paste into composite at vertical offset
            y_offset = row * single_frame_h
            composite.paste(frame_img, (0, y_offset))

            # Add label at bottom of this row band
            draw = ImageDraw.Draw(composite)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
                )
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            label = f"{seq['action']} | digits: {seq['digit_classes']} | t={t + 1:02d}"
            draw.text((5, y_offset + single_frame_h - 15), label, fill=255, font=font)

        gif_frames.append(composite)

    print(
        f"Generated {len(gif_frames)} composite frames at {fps} fps ({frame_duration_ms} ms/frame)."
    )

    # Save GIF
    print(f"Saving GIF to {output_path}...")
    gif_frames[0].save(
        output_path,
        format="GIF",
        append_images=gif_frames[1:],
        save_all=True,
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    print(
        f"GIF saved: {output_path} ({len(gif_frames)} frames, {gif_frames[0].size[0]}x{gif_frames[0].size[1]})"
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Moving MNIST sequence visualization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/moving-mnist-example.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of sequences (rows) to show",
    )
    parser.add_argument(
        "--seq-length", type=int, default=30, help="Frames per sequence"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Animation frames per second (default: 10, smoother than 5)",
    )
    parser.add_argument(
        "--upscale", type=int, default=4, help="Upscale factor for visibility"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None = random each run)",
    )
    args = parser.parse_args()

    create_animated_gif(
        output_path=args.output,
        num_sequences=args.num_sequences,
        seq_length=args.seq_length,
        fps=args.fps,
        upscale_factor=args.upscale,
        seed=args.seed,
    )
