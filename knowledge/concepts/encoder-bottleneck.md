---
title: "Encoder Bottleneck"
aliases: [spatial encoder limitation, rotation blindness, motion signal loss]
tags: [architecture-analysis, reservoir-computing, failure-mode, spatial-encoding]
sources:
  - "daily/2026-04-18.md"
created: 2026-04-18
updated: 2026-04-18
word_count: 450
---

# Encoder Bottleneck: The Root Cause of Spinning/Stationary Confusion

The **encoder bottleneck** is the fundamental architectural limitation preventing CluSTAR from distinguishing spinning (rotation) from stationary (still) digits. The spatial encoder produces nearly identical embeddings for rotating and still frames, losing the motion signal before it reaches the reservoir.

## The Problem

```
Spinning frame  → SpatialEncoder → 512-dim embedding ≈
Stationary frame → SpatialEncoder → 512-dim embedding   (nearly identical!)
```

A rotating digit and a still digit occupy the same pixels (same digit shape, same canvas position) — only the pixel arrangement differs slightly due to rotation. Through a random projection, these differences are diluted below the noise floor.

## Why the Reservoir Can't Recover the Signal

The reservoir receives spatial-only embeddings with no temporal awareness. It must discover "these pixels are rotating" from a sequence of nearly identical embeddings. If the embeddings for spinning and stationary look ~identical at the single-frame level, the reservoir has almost no signal to work with.

### Quantitative Evidence

Measured neuron-level statistics from actual reservoir states:

| Metric | Spinning | Stationary | Ratio | Separation? |
|--------|----------|-----------|-------|-------------|
| `std` | 0.170 | 0.111 | 1.54× | Moderate — overlapping distributions |
| `diff` | 0.032 | 0.020 | 1.62× | Moderate — better but still overlapping |

Only 64% of neurons show spinning std > 1.5× stationary std. The signal exists but is weak and noisy.

## Why Deeper Reservoirs Don't Help

A Deep Reservoir (2 layers, Layer 2 receiving `[s_t ; Δs_t]`) was implemented but **did not improve spinning detection** and **reduced accuracy** from 80.4% → 75.4%. Processing a weak signal through more nonlinear dynamics further obscures it rather than amplifying it.

## Signal Dilution Pipeline

Rotation at 60–180°/s (6–18°/frame) produces subtle pixel changes in a 28×28 digit. The signal is progressively diluted:

1. **Bilinear interpolation** — smooths rotation artifacts
2. **Random projection encoder** — doesn't amplify small differences; orthogonal projection preserves distances but doesn't enhance them
3. **Reservoir α-forgetting** — partially averages temporal oscillation
4. **Temporal aggregation** — mean/max statistics average out periodic signals

## Proposed Solutions (Ordered by Impact)

| Solution | Approach | Impact | Complexity |
|----------|----------|--------|------------|
| Frame differencing encoder | Feed `\|frame_t - frame_{t-1}\|` as second input channel | High — makes rotation explicit | Medium |
| Dual-stream routing | Spatial → slow clusters, temporal diff → fast clusters | High — leverages cluster structure | Medium |
| Higher omega | Increase rotation speed to 200-400°/s | Medium — larger per-frame changes | Low (config only) |
| Learnable encoder | Replace random projection with trained CNN | High but breaks reservoir paradigm | High |

The core insight: **no amount of reservoir layering can recover signal that was lost at the input**. The fix must happen at or before the encoder.
---ENDFILE---