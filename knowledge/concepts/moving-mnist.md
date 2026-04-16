---
title: "Moving MNIST"
aliases: ["Moving MNIST Dataset", "Bouncing Digits"]
tags: [computer-vision, benchmark, video-prediction, dynamics-modeling]
sources:
  - "daily/2026-04-16.md"  # Dataset design discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 280
---

# Moving MNIST

**Moving MNIST** is a synthetic video benchmark where MNIST digits move, bounce, and rotate within a frame. It tests **temporal dynamics modeling** without the complexity of real-world video.

## Dataset Generation

Each sequence:
- Starts with 1–2 MNIST digits (28×28 grayscale) on 64×64 canvas
- Each digit assigned: constant velocity, optional angular velocity
- Digits bounce off walls (elastic collision)
- Digit masks prevent overlap (occlusion handling)
- Fixed length: typically 20–30 frames

## Action Classes (CluSTAR Variant)

CluSTAR uses **4 labeled action categories**:

| Label | Description | Parameters |
|-------|-------------|------------|
| **Moving** | Linear translation only | vx, vy ≠ 0, ω=0 |
| **Spinning** | Pure rotation, no translation | vx=vy=0, ω ≠ 0 |
| **Collision** | Two digits, trajectories intersect | vx₁,vy₁ + vx₂,vy₂ |
| **Stationary** | No motion at all | vx=vy=ω=0 |

## Statistics

- Train: 50,000 sequences
- Validation: 10,000 sequences
- Test: 10,000 sequences
- Additional: 200,000 unlabeled for pre-training

Frame rate: 10–30 Hz (configurable). Pixel values: float32 normalized [0, 1].

## Why Moving MNIST?

**Pros:**
- Simple, reproducible, fast to generate
- No licensing issues (MNIST is public domain)
- Clear motion dynamics to model
- Easy to vary difficulty (single vs. double digits, speed, rotation)

**Cons:**
- Synthetic — domain gap to real videos
- Limited complexity (no texture, lighting, background)
- Action classes somewhat artificial

## Prior Art

Moving MNIST introduced by **Srivastava et al. (2015)** for video prediction. Extended by:
- **Mathieu et al. (2015)** — MC net for multi-frame prediction
- **Finn et al. (2016)** — stochastic video models
- **Kumar et al. (2020)** — rotation-invariant video models

CluSTAR uses a **custom variant** with 4 action categories and optional collision events.

## Data Format

Each sequence returns:
```python
{
  'frames': Tensor[T, 64, 64],      # grayscale video
  'actions': int (0–3),             # class label
  'digits': List[int],              # which MNIST classes
  'velocities': Tensor[2],          # [vx, vy] per digit
  'rotations': Tensor[1],           # angular velocity ω
}
```

## Loading in CluSTAR

```python
from clustar.data.generator import MovingMNISTGenerator

gen = MovingMNISTGenerator(mnist_pkl_path="mnist.pkl")
seq = gen.generate_sequence(action="spinning", seq_length=30)
```

Generator uses digit templates from local `mnist.pkl` file (no internet download).
