---
title: "Spatial Encoder"
aliases: [random projection encoder, static encoder]
tags: [feature-engineering, dimensionality-reduction, reservoir-computing]
sources:
  - "daily/2026-04-16.md"  # Spatial encoder design
created: 2026-04-16
updated: 2026-04-16
word_count: 200
---

# Spatial Encoder

The **Spatial Encoder** in CluSTAR performs fixed, non-learned dimensionality reduction from raw image pixels to a compact feature vector fed into the reservoir.

## Purpose

Raw MNIST frames: 28×28 = 784 dimensions → too high for small reservoir (would dilute dynamics). Encoder projects to **D=128** (or 256) dims, injecting spatial bias while remaining fixed (no training).

## Implementation: Random Orthogonal Projection

```
u(t) = R · flatten(I_t)
```

Where:
- **R** ∈ ℝ^(D×784) is a random matrix with orthogonal columns (R^T · R ≈ I)
- Orthogonality preserves Euclidean distances → no distortion
- Sampled once at initialization, never updated
- Analogy to **Random Kitchen Sinks** (Rahimi & Recht, 2007)

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| Random projection (chosen) | No training, fast, preserves structure | Fixed; may not align with task |
| PCA (pre-computed) | Optimal linear compression | Needs MNIST statistics; loses nonlinear info |
| Fixed 1×1 conv | Spatial→channel mixing; CNN-like | Still linear; requires hand-design |
| Small fixed CNN | Inductive bias (edges, corners) | More parameters; still not learned |

## Input to Reservoir

After encoding, feature vector `u(t)` is split into streams:
```
u = [u_pos; u_vel; u_rot; u_shape]  # concatenated
```
Each stream is routed to specific reservoir clusters.

## Psychology of Fixed Encoder

Using a fixed random encoder is common in:
- **Random Features** (Rahimi & Recht) for kernel approximation
- **ESN literature** (input weights typically random)
- **MLP-Mixer** (fixed linear projection per modality)

It offloads spatial summarization to reservoir dynamics rather than learned features, reducing trainable parameters to **zero** in encoder.
