---
title: "Clustered Reservoir"
aliases: [structured reservoir, small-world reservoir, functional clustering]
tags: [reservoir-computing, architecture-design, graph-neural-networks]
sources:
  - "daily/2026-04-16.md"  # Clustered topology design
created: 2026-04-16
updated: 2026-04-16
word_count: 300
---

# Clustered Reservoir

The **Clustered Reservoir** is the core structural innovation in CluSTAR. Instead of random Erdős–Rényi connectivity, neurons are partitioned into functionally specialized clusters with **small-world topology**.

## Topology Design

```
10 clusters × 100 neurons each (total N=1000)
```

**Within-cluster connectivity**: dense (p=0.3) — allows rich intra-group dynamics
**Between-cluster connectivity**: sparse (p=0.02) — long-range integration

This creates a **small-world graph** (high clustering coefficient + short average path length), balancing:
- **Specialization** (within-cluster computation)
- **Integration** (between-cluster communication)

## Functional Specialization

Each cluster is allocated to a motion feature domain:

| Clusters | Feature | Input Stream | Role |
|----------|---------|--------------|------|
| 1–3 | Position (x, y) | `u_pos` (centroid coords) | Track digit location |
| 4–6 | Velocity (vx, vy) | `u_vel` (optical flow) | Encode motion vectors |
| 7–8 | Rotation (θ, ω) | `u_rot` (angle, angular velocity) | Capture spinning |
| 9–10 | Digit Identity | `u_shape` (Hu moments) | Recognize digit class |

This **input routing** ensures each cluster's recurrent dynamics focus on a specific motion primitive.

## Small-World vs. Random

**Erdős–Rényi random graph** (vanilla ESN):
- p(edge) = same for all neuron pairs
- No community structure
- Lower clustering coefficient

**Watts–Strogatz small-world** (CluSTAR):
- Start with ring lattice (each neuron connected to k nearest)
- Rewire each edge with probability p_rewire
- Results in **high clustering + short path lengths**

**Benefits**:
- Clusters act as **feature detectors** (position cluster becomes velocity-sensitive)
- Modular structure aids interpretability
- Robust to neuron failures (damage one cluster → only that feature degrades)

## Spectral Radius per Cluster

Each cluster group can have its own **spectral radius** ρ_c ∈ [0.7, 0.99], controlling echo depth:
- **Position clusters**: ρ≈0.7 — short memory (position changes quickly)
- **Velocity clusters**: ρ≈0.85 — medium memory (momentum over few frames)
- **Rotation clusters**: ρ≈0.9 — longer memory (rotation is periodic, needs context)
- **Identity clusters**: ρ≈0.95 — longest memory (digit shape stable)

## Implementation

Reservoir weight matrix `W_res` is block-structured:
```
W_res = [ C₁₁   C₁₂   C₁₃   C₁₄ ]
        [ C₂₁   C₂₂   C₂₃   C₂₄ ]
        [ C₃₁   C₃₂   C₃₃   C₃₄ ]
        [ C₄₁   C₄₂   C₄₃   C₄₄ ]
```
Each block C_ij ∈ ℝ^(n_i×n_j) has density based on cluster relationship.

## Ablation Study

Key ablations:
- **No clustering** (fully random connectivity) → tests value of modularity
- **Uniform spectral radius** → tests value of heterogeneous timescales
- **Random input routing** (all features to all clusters) → tests functional specialization

Expected: full CluSTAR > all ablations.
