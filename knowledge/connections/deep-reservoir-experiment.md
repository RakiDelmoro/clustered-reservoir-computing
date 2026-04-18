---
title: "Deep Reservoir Experiment: Implementation and Results"
source: "daily/2026-04-18.md"
tags: [deep-reservoir, experiment, architecture, failure-analysis, clustar]
---

# Deep Reservoir Experiment

An experiment implementing a 2-layer deep reservoir for CluSTAR to improve spinning vs stationary discrimination. The experiment **failed to improve results** — accuracy dropped from 80.4% to 75.4% and spinning detection remained at 0%.

## Architecture

```
Frame t (64×64)
      ↓
Spatial Encoder (4096 → 512, fixed random projection)
      ↓ encoded_t (512-dim)
      ↓
Reservoir Layer 1 (1000 neurons, 10 clusters, mixed timescales)
      ↓ s_t (1000-dim state)
      ↓         ↘
      ↓          Δs_t = s_t - s_{t-1} (1000-dim change signal)
      ↓         ↙
      ↓  [s_t ; Δs_t] (2000-dim concatenated)
      ↓
Reservoir Layer 2 (500 neurons, 5 clusters, fast-biased timescales)
      ↓ z_t (500-dim state)
      ↓
Temporal Aggregation (per cluster, per layer):
  Layer 1: [first, last, mean, max] × 10 clusters → 4000-dim
  Layer 2: [first, last, mean, max] × 5 clusters  → 2000-dim
      ↓
Concat → 6000-dim → Ridge Regression Readout (6000 → 3)
```

## Design Rationale

| Layer | Input | Captures | Timescales |
|-------|-------|----------|------------|
| Layer 1 | Raw spatial features | Position, digit shape, appearance | Mixed (0.2–1.0) |
| Layer 2 | `[s_t ; Δs_t]` — state AND state change | Motion patterns, rotation, stillness | Fast-biased (0.6–1.0) |

Layer 2 receives explicit temporal change signal (Δs_t). Theory: Δs_t ≈ 0 for stationary, periodic for spinning, constant for moving. Layer 2's recurrent dynamics would then capture the *pattern* of change.

## Key Design Decisions

1. **Layer 2 input**: `[s_t ; Δs_t]` (both state and change) — provides both "what is" and "what's changing"
2. **Layer 2 size**: 500 neurons (half of Layer 1) — temporal features need less capacity
3. **Layer 2 clusters**: 5 (half of Layer 1) — maintains cluster structure
4. **Layer 2 timescales**: fast-biased (α ∈ [0.6, 1.0]) — track rapid changes
5. **Aggregation**: only `[first, last, mean, max]` per layer (no std/diff at this stage)
6. **Feature dim**: 6000 (4000 from Layer 1 + 2000 from Layer 2)

## Results

| Metric | Single-Layer (80.4%) | Deep Reservoir (75.4%) | Change |
|--------|---------------------|----------------------|--------|
| Test accuracy | 80.4% | 75.4% | **-5%** |
| Moving detection | Good (frames 19-60) | Better (frames 5-60, faster lock-on) | + |
| Spinning detection | Never (0%) | Never (0%) | Same |
| Stationary detection | Good (frames 173+) | Good (frames 182+, 72% confidence) | Similar |

## Why It Failed

1. **Δs_t signal is too weak**: Spinning vs stationary Δs_t has only a 1.54× ratio — adding more nonlinear processing obscures rather than amplifies
2. **More dimensions, same signal**: 6000-dim feature space has more noise relative to signal; ridge regression must fit more weights with same number of samples
3. **Fast-biased timescales may be too aggressive**: Loss of context in Layer 2
4. **The bottleneck is upstream**: The spatial encoder produces nearly identical embeddings for spinning and stationary frames — no reservoir architecture can recover lost input signal

## Key Lesson

> **The bottleneck is not in the reservoir architecture — it's in the encoder.** No amount of reservoir layering can recover signal that was lost at the input. The fix must happen at or before the spatial encoder (e.g., frame differencing as a second input channel).

## Related

- [[concepts/deep-echo-state-network]] — general Deep ESN concept
- [[concepts/encoder-bottleneck]] — root cause analysis
- [[concepts/clustar]] — main architecture
---ENDFILE---