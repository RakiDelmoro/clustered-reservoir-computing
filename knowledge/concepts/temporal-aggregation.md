---
title: "Temporal Aggregation"
aliases: [sequence pooling, temporal pooling, temporal feature summarization, sliding window aggregation]
tags: [feature-engineering, temporal-dynamics, readout-layer, reservoir-computing]
sources:
  - "daily/2026-04-16.md"
  - "daily/2026-04-18.md"
created: 2026-04-16
updated: 2026-04-18
word_count: 480
---

# Temporal Aggregation

**Temporal aggregation** converts a temporal sequence of reservoir states `x(0), x(1), ..., x(T-1)` into a fixed-length vector for **sequence-level** predictions (e.g., action classification). It determines how temporal information is compressed before the readout layer.

## Problem Statement

Reservoir state sequence: `X_seq ∈ ℝ^(T×N)` where T = sequence length, N = reservoir size.

Readout needs: `X_fixed ∈ ℝ^D` where D is fixed.

**Aggregation function f**: `ℝ^(T×N) → ℝ^D`

## CluSTAR Aggregation: 6 Statistics (Updated 2026-04-18)

Per cluster, computed over the sequence of reservoir states belonging to that cluster:

| Statistic | Formula | Captures | Best For |
|-----------|---------|----------|----------|
| **first** | `states[:, 0, :]` | Initial condition | Context at start |
| **last** | `states[:, -1, :]` | Final state | Current/ending state |
| **mean** | `states.mean(dim=1)` | Average activation | Overall activity level |
| **max** | `states.max(dim=1)[0]` | Peak response | Salient events |
| **std** | `states.std(dim=1)` | Temporal variability | Oscillation vs constancy |
| **diff** | `(states[:,1:,:] - states[:,:-1,:]).abs().mean(dim=1)` | Rate of change | Motion vs stillness |

Feature dim: `6 × reservoir_size` per layer.

**Why std and diff were added**: The original 4 stats (`[first, last, mean, max]`) cannot distinguish a constant signal (stationary) from a periodic oscillation (spinning) because `mean` averages out oscillations. `std` directly measures temporal spread (stationary ≈ 0, spinning > 0), while `diff` measures frame-to-frame change rate (stationary ≈ 0, moving >> 0, spinning > 0).

### Intuition: Why Mean Destroys Periodicity

| Signal | Values | mean | std | diff |
|--------|--------|------|-----|------|
| Stationary | [0.1, 0.1, 0.1, ...] | 0.1 | **≈0** | **≈0** |
| Spinning | [0.1, 0.3, 0.5, 0.3, 0.1, -0.1, ...] | **0.1** | **0.2** | **0.15** |
| Moving | [0.1, 0.2, 0.3, 0.4, ...] | 0.4 | 0.2 | 0.1 |

Stationary and spinning have nearly the same `mean` (0.1 vs 0.1), but `std` and `diff` clearly separate them.

### Reality Check

Real neuron statistics show weaker separation than the idealized example:
- Spinning vs stationary `std` ratio: **1.54×** (overlapping distributions)
- Spinning vs stationary `diff` ratio: **1.62×** (better but still overlapping)
- Only 64% of neurons show spinning std > 1.5× stationary std

The signal exists but is weak — the fundamental problem is upstream in the encoder (see [[concepts/encoder-bottleneck]]).

## Sliding Window Aggregation (For Inference)

During autoregressive inference, using a growing list of states causes feature distribution drift. Fixed with `collections.deque(maxlen=seq_length)`:

- Frames 1–60: window grows (matches training warmup)
- Frame 61+: oldest states drop off, window always has last 60 states
- Ensures aggregation distribution matches training conditions

See [[concepts/autoregressive-inference]] for full details.

## Historical Aggregation Methods (From Architecture Exploration)

| Method | Formula | Output dim | What it captures | Pros | Cons |
|--------|---------|------------|------------------|------|------|
| **Mean pooling** | `mean(X_seq, dim=0)` | N | Average activation | Simple, stable | Dilutes transient peaks |
| **Last state** | `X_seq[-1, :]` | N | Final context only | Minimal | Misses early cues |
| **Max pooling** | `max(X_seq, dim=0)` | N | Peak responses | Highlights salient events | Sensitive to outliers |
| **Concat first+last** | `[X_seq[0,:]; X_seq[-1,:]]` | 2N | Start + end state | Captures displacement | Ignores interior |
| **Cat last 3** | `[X_seq[-3:,:]]` flatten | 3N | Recent trajectory | Short-term context | Still misses long-term |
---ENDFILE---