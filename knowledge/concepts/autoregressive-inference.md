---
title: "Autoregressive Inference"
aliases: [frame-by-frame inference, online inference, streaming inference, sliding window aggregation]
tags: [inference, reservoir-computing, temporal-dynamics, real-time]
sources:
  - "daily/2026-04-18.md"
created: 2026-04-18
updated: 2026-04-18
word_count: 420
---

# Autoregressive Inference

**Autoregressive inference** runs the CluSTAR model frame-by-frame on a video stream, carrying reservoir state forward between frames rather than processing fixed-length sequences independently. This enables real-time prediction but introduces the **unbounded aggregation problem**.

## Pipeline

```
Frame t → SpatialEncoder → 512-dim embedding
         → ClusteredReservoir(prev_state) → 1000-dim state (carry forward)
         → Append to sliding window (maxlen=seq_length)
         → Per-cluster aggregation over window
         → Standardize with saved feat_mean/feat_std
         → ActionReadout → logits → argmax → prediction
```

The reservoir state is inherently recurrent — each frame's state depends on all previous frames via α-forgetting. No need to reprocess previous frames.

## The Unbounded Aggregation Problem

The initial implementation appended every reservoir state to a growing list, then aggregated all states with `[first, last, mean, max]`. This caused predictions to "stick" to the first action detected:

- Frame 100: aggregation over states 1→100 — old action dominates
- Feature distribution drifts away from training distribution (trained on fixed 60-frame sequences)
- Model predicted "moving" for almost the entire video regardless of actual action

## Fix: Sliding Window Aggregation

Replace unbounded `list` with `collections.deque(maxlen=seq_length)`:

| Frame | Window Contents | Behavior |
|-------|----------------|----------|
| 1–59 | states[0:t+1], growing | Warmup — matches training warmup |
| 60 | states[0:60] — full window | Exact match to training distribution |
| 61+ | states[t-59:t+1] — sliding | Oldest states drop off automatically |

**Effect on action transitions**: As old-action states exit the window, aggregation statistics shift toward the new action. With `seq_length=60`, an action transition fully registers within 60 frames.

## Reservoir State vs Aggregation Window

The **reservoir state itself** still carries influence from all past frames (autoregressive with α-forgetting). The sliding window only bounds the **aggregation step** (first, last, mean, max, std, diff). There are two independent forgetting mechanisms:

1. **α-forgetting** (reservoir): exponential decay of state influence, always active
2. **Sliding window** (aggregation): hard cutoff at `seq_length` frames, bounds feature distribution

## Cold-Start Behavior

Early frames (1–10) have very few states in the window. Predictions are noisy and unreliable. The model was trained on 60-frame sequences, so predictions stabilize around frame 30–40 when the window has accumulated enough states.

## Implementation

Script: `clustar/scripts/run_inference.py`

Key components:
- Loads model from checkpoint (encoder + reservoir reconstructed from config, readout weights loaded)
- Reads video via `imageio.mimread()`
- Ground truth labels from metadata JSON (generated alongside test video)
- Outputs per-frame prediction table with target, predicted action, and confidence
---ENDFILE---