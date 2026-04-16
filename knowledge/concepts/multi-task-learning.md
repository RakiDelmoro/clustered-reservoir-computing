---
title: "Multi-Task Learning"
aliases: [multi-objective learning, joint training]
tags: [machine-learning, training-paradigm, representation-learning]
sources:
  - "daily/2026-04-16.md"  # Multi-task pretext training discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 260
---

# Multi-Task Learning (MTL)

**Multi-task learning** trains a single model on multiple related objectives simultaneously, encouraging the shared representation to be useful across all tasks. In CluSTAR, MTL is used during **self-supervised pre-training**.

## Multi-Task in CluSTAR Pre-training

**Shared backbone**: spatial encoder + structured reservoir
**Task-specific heads** (all linear readouts):
1. Frame prediction head (784-dim regression)
2. Temporal order head (binary classification)
3. Speed regression head (scalar regression)
4. Segmentation head (binary mask)
5. Rotation contrast head (binary classification)

**Loss**:
```
L_total = λ₁·L_frame + λ₂·L_order + λ₃·L_speed + λ₄·L_seg + λ₅·L_rot
```

Weights λ are tuned so tasks are balanced (no single task dominates).

## Benefits

- **Richer representations**: Reservoir must encode features useful for diverse temporal reasoning tasks
- **Regularization**: Joint training prevents overfitting to any single pretext task
- **Efficiency**: Single forward pass through reservoir, multiple heads trained jointly
- **Knowledge transfer**: Learning one task (e.g., temporal order) helps others (e.g., frame prediction)

## Challenges

- **Task interference**: gradients from conflicting objectives can destabilize training
- **Hyperparameter tuning**: loss weights λ need tuning
- **Gradient scale differences**: regression vs classification losses have different magnitudes → careful normalization needed
- **Shared capacity limits**: reservoir size must accommodate all task requirements

## Why MTL Works for Dynamics Pre-training

Our five pretext tasks target **complementary aspects** of motion understanding:
- **What** changes (frame prediction → pixel reconstruction)
- **When** changes occur (temporal order → causality)
- **How fast** (speed regression → motion magnitude)
- **What is object** (segmentation → foreground extraction)
- **Which way** (rotation contrast → orientation invariance)

No single task captures all these; together they force reservoir to build a **general dynamics world model**.

## Closed-Form MTL via Ridge Regression

Unlike deep MTL (gradient-based), CluSTAR's MTL is **non-iterative**:
- Collect states X once
- Solve separate ridge problems: `W_i = Y_i · X^T · (X·X^T + λ_i I)^{-1}`
- Concatenate W_i into multi-task readout

This is possible because reservoir states are fixed; no gradient interference across tasks during training.
