---
title: "CluSTAR"
aliases: ["Clustered Spatio-Temporal Reservoir", "Clustered ESN"]
tags: [reservoir-computing, architecture-innovation, world-model, embedded-ai, moving-mnist]
sources:
  - "daily/2026-04-16.md"  # Entire conversation about CluSTAR design
created: 2026-04-16
updated: 2026-04-16
word_count: 550
---

# CluSTAR: Clustered Spatio-Temporal Reservoir

**CluSTAR** is a novel structured reservoir computing architecture designed for **visual world models on embedded devices**. It extends classical Echo State Networks with three key innovations: (1) **clustered reservoir topology**, (2) **multi-timescale neuron dynamics**, and (3) **spatial-encoded input routing**.

## Core Innovations

### 1. Clustered Reservoir Topology

Instead of random Erdős–Rényi connectivity, neurons are partitioned into **functionally specialized clusters** with small-world connectivity:

- **10 clusters × 100 neurons** (total N=1000)
- **Within-cluster**: dense connectivity (p=0.3)
- **Between-cluster**: sparse long-range links (p=0.02) — Watts-Strogatz small-world graph
- Each cluster learns to track specific motion features (position, velocity, rotation, digit identity)

### 2. Multi-Timescale Dynamics

Neurons within each cluster share a **leaking rate α** determining memory persistence:

- **Fast neurons** (α≈0.3): short-term, high-frequency motion
- **Medium neurons** (α≈0.6): intermediate dynamics
- **Slow neurons** (α≈0.9): long-term context, digit identity

This heterogeneity allows reservoir to capture dynamics across multiple temporal horizons simultaneously.

### 3. Spatial Encoder + Input Routing

**Before** reservoir: flattened 784-dim (28×28) frame → random orthogonal projection (784→128).

**Input routing**: encoded features split into streams based on semantics:

- `u_pos`: digit centroid coordinates → position clusters (1–3)
- `u_vel`: optical flow magnitude/direction → velocity clusters (4–6)
- `u_rot`: rotation angle (via PCA) → rotation clusters (7–8)
- `u_shape`: digit morphology (Hu moments) → identity clusters (9–10)

Each stream connects only to its designated cluster group, encouraging functional specialization.

## Full Architecture

```
Moving MNIST frames (28×28) 
    → Spatial Encoder (random projection, 784→128)
    → Feature Streams (pos, vel, rot, shape)
    → Clustered Reservoir (10 clusters × 100 neurons)
    → Multi-Task Readout Heads:
        • Head 1: Future frame prediction (world model, 784-dim)
        • Head 2: Action classification (4-class softmax)
        • Head 3: Memory consistency (optional)
```

## Training Pipeline

**Phase 1: Self-Supervised Pre-training** (200K unlabeled sequences)
- Run reservoir once, collect states
- Train all readout heads jointly via **multi-task ridge regression**
- Pretext tasks: frame prediction, temporal order verification, speed regression, segmentation mask prediction, rotation contrast

**Phase 2: Fine-tuning** (50K labeled sequences)
- Freeze reservoir + spatial encoder
- Train only action classifier readout via ridge regression (λ=1e-4)
- Evaluates quality of pre-trained dynamics features

## Why CluSTAR Works

1. **Structured inductive bias** — clusters align with motion semantics
2. **Multi-timescale hierarchy** — captures fast bouncing + slow rotation concurrently
3. **Parameter efficiency** — only ~10K trainable readout parameters (vs. millions in LSTM)
4. **Embedded-friendly** — no BPTT, fixed weights, linear readout training
5. **Interpretability** — cluster activations can be visualized to understand what motion aspect each group tracks

## Baselines

Compared against:
1. **Vanilla ESN** — unstructured random reservoir
2. **1-Layer LSTM** (128 units) — gradient-based recurrent baseline
3. **Random Reservoir + Augmentation** — same architecture but trained with data augmentation only
4. **CluSTAR (no pre-train)** — ablate self-supervised pre-training

## Expected Benefits

- Higher action classification accuracy than vanilla ESN
- Better world model predictions (lower frame MSE)
- Faster convergence due to pre-trained features
- Cluster-based interpretability of learned dynamics

## Applications

- **Embedded robotics** — real-time motion prediction on microcontroller
- **Edge video understanding** — action recognition without cloud
- **Neuromorphic computing** — potential mapping to spiking LSM variant
- **Dynamical system identification** — general time-series modeling

## Paper Reference

Target venue: **NeurIPS / ICML / IROS** (robotics + ML intersection). Novelty claim: "first structured reservoir with feature routing and multi-timescale clusters for visual world models on resource-constrained devices."
