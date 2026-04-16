---
title: "CluSTAR Architecture Components"
source: "daily/2026-04-16.md"
tags: [clustar, architecture, component-relationships]
---

# CluSTAR: How Components Interact

CluSTAR's innovations are not independent — they form a **coherent pipeline** where each component's output feeds the next, creating emergent properties greater than the sum of parts.

## Data Flow and Dependencies

```
Raw Frames (28×28)
    ↓
[1] Spatial Encoder (Random Projection 784→128)
    ↓ produces u(t) ∈ ℝ¹²⁸
    ↓
[2] Input Routing (Split into 4 streams)
    ↓ u_pos, u_vel, u_rot, u_shape
    ↓
[3] Clustered Reservoir (10 clusters × 100 neurons)
    ↓ x(t+1) = tanh(W_in·u + W_res·x(t))
    ↓
[4] Multi-Timescale Neurons (within clusters)
    ↓ fast (α=0.3), medium (0.6), slow (0.9)
    ↓
[5] Reservoir States x(t) ∈ ℝ¹⁰⁰⁰
    ↓
[6] Multi-Task Readout Heads
    ├─ Frame predictor (world model)
    ├─ Action classifier (downstream)
    └─ Optional: consistency head
```

## Component Interactions

### 1. Spatial Encoder → Input Routing
The encoder reduces pixels to 128-dim vector **without** structure; routing imposes structure. Encoder must preserve enough information so that `u_pos` carries position data, `u_vel` carries motion, etc. Random orthogonal projection approximately preserves distances → streams remain meaningful.

### 2. Input Routing → Clustered Reservoir
Routing **pre-assigns** semantic roles to clusters. Position stream only excites clusters 1–3. Over many timesteps, recurrent dynamics within those clusters become **fine-tuned** to position tracking (via Hebbian-like implicit effect of repeated activation).

### 3. Clustered Reservoir → Multi-Timescale
Within each cluster, fast/medium/slow neurons form a **hierarchy of temporal integration**:
- Fast neurons detect instantaneous changes
- Medium integrate over ~0.5 seconds
- Slow neurons maintain persistent digit identity

The readout can then pool across timescales for different tasks:
- Frame prediction uses **all** timescales
- Action classification may rely more on **slow** (identity) or **fast** (motion) depending on label

### 4. Multi-Timescale → Readout Heads
The multi-timescale reservoir state `x(t)` is **high-dimensional (1000)** but structured:
- Dimensions 0–99: fast pos neurons
- 100–199: medium pos, etc.

Readout weights `W_action` implicitly learn which timescale cluster matters for each action:
- "spinning" likely uses rotation clusters (slow, persistent rotation signal)
- "moving" uses velocity clusters (medium timescale)
- "collision" uses fast transient spikes when digits overlap

## Emergent Properties

The combination yields:
1. **Functional specialization** (clusters develop tunable selectivity)
2. **Multi-scale temporal context** (single forward pass captures short & long patterns)
3. **Sparse efficient readouts** (only 10K trainable params vs. millions in CNN+LSTM)
4. **Interpretability** (visualize cluster activations to understand what motion feature triggered classification)

## Ablation Implications

Removing any component breaks the chain:
- **No spatial encoder** → routing meaningless (can't split features)
- **No routing** → clusters must discover semantics from scratch (harder)
- **No clustering** → multi-timescale less meaningful (no functional grouping)
- **Uniform α** → loses hierarchical temporal abstraction

Full CluSTAR = all components **synergize**.
