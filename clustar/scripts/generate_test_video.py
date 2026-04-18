import sys
import os
import json
import numpy as np
import torch
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.generator import MovingMNISTGenerator
from torchvision.transforms.functional import rotate as tv_rotate
import torchvision.transforms

ACTIONS = ["moving", "spinning", "stationary"]
ACTION_PARAMS = {
    "moving": {"v_range": (2, 6), "rotation": False, "omega_range": (0, 0)},
    "spinning": {"v_range": (0, 0.1), "rotation": True, "omega_range": (60, 180)},
    "stationary": {"v_range": (0, 0), "rotation": False, "omega_range": (0, 0)},
}


def sample_action_params(action, rng):
    p = ACTION_PARAMS[action]
    speed = rng.uniform(*p["v_range"])
    angle = rng.uniform(0, 2 * np.pi)
    vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])
    omega = 0.0
    if p["rotation"]:
        omega = rng.uniform(*p["omega_range"]) * (1 if rng.rand() > 0.5 else -1)
    return vel, omega


def render_digit(canvas, d_info, canvas_size, img_size):
    digit_img = d_info["image"]
    if abs(d_info["angle"]) > 0.1:
        digit_img = tv_rotate(
            digit_img,
            -d_info["angle"],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            expand=False,
        )
    x, y = int(d_info["pos"][0]), int(d_info["pos"][1])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas_size, x + img_size), min(canvas_size, y + img_size)
    dx1, dy1 = max(0, -x), max(0, -y)
    dx2 = img_size - max(0, (x + img_size) - canvas_size)
    dy2 = img_size - max(0, (y + img_size) - canvas_size)
    if x2 > x1 and y2 > y1:
        canvas[0, y1:y2, x1:x2] = torch.maximum(
            canvas[0, y1:y2, x1:x2], digit_img[0, dy1:dy2, dx1:dx2]
        )


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def main():
    seed = 42
    rng = np.random.RandomState(seed)

    canvas_size = 64
    img_size = 28
    fps = 10.0
    steady_frames = 60
    transition_frames = 15

    generator = MovingMNISTGenerator(
        img_size=img_size, canvas_size=canvas_size, seed=seed
    )

    digit_class = rng.randint(0, 10)
    digit_img = generator.digit_images[digit_class]

    action_vels = []
    action_omegas = []
    for action in ACTIONS:
        vel, omega = sample_action_params(action, rng)
        action_vels.append(vel)
        action_omegas.append(omega)

    d_info = {
        "class": digit_class,
        "image": digit_img.clone(),
        "pos": np.array(
            [
                float(rng.randint(0, canvas_size - img_size)),
                float(rng.randint(0, canvas_size - img_size)),
            ],
            dtype=float,
        ),
        "vel": action_vels[0].copy(),
        "angle": 0.0,
        "omega": action_omegas[0],
    }

    frames = []
    frame_labels = []
    schedule = []

    for action_idx in range(len(ACTIONS)):
        cur_vel = action_vels[action_idx]
        cur_omega = action_omegas[action_idx]
        action = ACTIONS[action_idx]

        if action_idx == 0:
            d_info["vel"] = cur_vel.copy()
            d_info["omega"] = cur_omega

        start_frame = len(frames)
        for _ in range(steady_frames):
            d_info["pos"] += d_info["vel"] * (1.0 / fps)
            d_info["angle"] += d_info["omega"] * (1.0 / fps)
            d_info["angle"] %= 360
            for dim in [0, 1]:
                if d_info["pos"][dim] < 0:
                    d_info["pos"][dim] = 0
                    d_info["vel"][dim] *= -1
                elif d_info["pos"][dim] > canvas_size - img_size:
                    d_info["pos"][dim] = canvas_size - img_size
                    d_info["vel"][dim] *= -1

            canvas = torch.zeros(1, canvas_size, canvas_size)
            render_digit(canvas, d_info, canvas_size, img_size)
            frames.append((canvas[0].numpy() * 255).astype(np.uint8))
            frame_labels.append(action)

        schedule.append(
            {"action": action, "start": start_frame, "end": len(frames) - 1}
        )

        if action_idx < len(ACTIONS) - 1:
            next_action = ACTIONS[action_idx + 1]
            next_vel = action_vels[action_idx + 1]
            next_omega = action_omegas[action_idx + 1]

            transition_start_vel = d_info["vel"].copy()
            transition_start_omega = d_info["omega"]

            t_start = len(frames)
            for f in range(transition_frames):
                t = smoothstep((f + 1) / transition_frames)
                d_info["vel"] = lerp(transition_start_vel, next_vel, t)
                d_info["omega"] = lerp(transition_start_omega, next_omega, t)

                d_info["pos"] += d_info["vel"] * (1.0 / fps)
                d_info["angle"] += d_info["omega"] * (1.0 / fps)
                d_info["angle"] %= 360
                for dim in [0, 1]:
                    if d_info["pos"][dim] < 0:
                        d_info["pos"][dim] = 0
                        d_info["vel"][dim] *= -1
                    elif d_info["pos"][dim] > canvas_size - img_size:
                        d_info["pos"][dim] = canvas_size - img_size
                        d_info["vel"][dim] *= -1

                canvas = torch.zeros(1, canvas_size, canvas_size)
                render_digit(canvas, d_info, canvas_size, img_size)
                frames.append((canvas[0].numpy() * 255).astype(np.uint8))
                frame_labels.append(next_action)

            schedule.append(
                {
                    "action": f"{action}→{next_action}",
                    "start": t_start,
                    "end": len(frames) - 1,
                }
            )

    output_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    video_path = os.path.join(output_dir, "test.mp4")
    meta_path = os.path.join(output_dir, "test_meta.json")

    imageio.mimwrite(video_path, frames, fps=fps)

    metadata = {
        "fps": fps,
        "total_frames": len(frames),
        "steady_frames": steady_frames,
        "transition_frames": transition_frames,
        "actions": ACTIONS,
        "schedule": schedule,
        "frame_labels": frame_labels,
        "digit_class": digit_class,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total = len(frames)
    print(
        f"Saved test.mp4 ({total} frames, {fps} fps, {total / fps:.1f}s) to {video_path}"
    )
    print(f"Saved test_meta.json to {meta_path}")
    print(
        f"Digit {digit_class}: {steady_frames} steady + {transition_frames} transition per action"
    )
    print(f"Actions: {' → '.join(ACTIONS)}")


if __name__ == "__main__":
    main()
