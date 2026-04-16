---
title: "Input Routing"
aliases: [feature routing, stream partitioning, functional specialization]
tags: [architecture-design, inductive-bias, reservoir-computing]
sources:
  - "daily/2026-04-16.md"  # Input routing to clusters discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 220
---

# Input Routing

**Input routing** in CluSTAR partitions the encoded feature vector into **semantic streams**, each connected exclusively to a specific functional cluster group in the reservoir.

## Motivation

In vanilla ESN, all input features connect to all reservoir neurons uniformly. This lacks inductive bias: position features and rotation features get mixed together before clustering emerges. Input routing **pre-structures** connectivity based on domain knowledge.

## Implementation

After spatial encoder produces `u(t) ∈ ℝ^D` (D=128), split:

```
u_pos  = u[0:32]   → Clusters 1–3 (position)
u_vel  = u[32:64]  → Clusters 4–6 (velocity)
u_rot  = u[64:80]  → Clusters 7–8 (rotation)
u_shape= u[80:128] → Clusters 9–10 (identity)
```

Connections: W_in is block-diagonal, each block connects one stream to its cluster(s). No cross-connection between streams at input layer (they can still interact through reservoir feedback).

## Feature Extraction per Stream

| Stream | Computed From | Dimensions | Semantic |
|--------|---------------|------------|----------|
| `u_pos` | Digit centroid (x, y) plus bias | 32 | Where digit is |
| `u_vel` | Frame differencing + coarse optical flow | 32 | How fast moving |
| `u_rot` | PCA-based orientation angle & angular velocity | 16 | Spinning direction |
| `u_shape` | Hu moment invariants (7) + expanded | 48 | Digit identity (0–9) |

These are **hand-crafted features** over raw pixels, injecting vision priors before reservoir.

## Why Routing Helps

1. **Functional specialization** — position cluster neurons don't waste capacity on rotation features
2. **Reduced interference** — streams evolve semi-independently
3. **Interpretability** — we know which cluster processes which motion primitive
4. **Parameter efficiency** — W_in is block-structured, fewer active connections

## Alternatives

- **Full connectivity** (vanilla ESN) — no routing
- **Soft routing** (attention over clusters) — learnable but expensive
- **Cluster-blind reservoir** — no routing, rely on reservoir to self-organize

**CluSTAR chooses hard-coded routing** based on motion semantics. A data-driven variant could learn routing weights but loses embedded efficiency.

## Connection to Mixture of Experts

Input routing resembles **Mixture of Experts (MoE)** where feature streams are assigned to expert modules (clusters). Unlike MoE's gating network, routing here is deterministic by feature type.
