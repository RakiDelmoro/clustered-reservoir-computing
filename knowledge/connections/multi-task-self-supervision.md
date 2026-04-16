---
title: "Self-Supervised Pre-training Meets Multi-Task Learning"
source: "daily/2026-04-16.md"
tags: [self-supervised-learning, multi-task-learning, pretext-tasks, representation-learning]
---

# Multi-Task Self-Supervision for Dynamics

CluSTAR's pre-training phase combines **self-supervised learning** (no labels) with **multi-task learning** (multiple objectives) to learn a general visual dynamics representation.

## Why Both?

**Self-supervision alone** could use any single pretext task (e.g., frame prediction). But a single task might bias toward specific features (e.g., frame prediction focuses on pixel reconstruction, not semantic motion).

**Multi-task alone** (supervised multi-task) would require multiple labeled datasets, expensive.

**Multi-task self-supervision** gets the best of both: many free learning signals from unlabeled video, each task targeting a different facet of dynamics understanding.

## The Five Pretext Tasks

Each taskforces reservoir to encode different motion semantics:

| Task | Input | Target | Forces Encoding Of |
|------|-------|--------|-------------------|
| **Frame Prediction** | frames t-K..t | frame t+1 | Pixel-level dynamics |
| **Temporal Order** | two windows of frames | which came first? | Arrow of time, causality |
| **Speed Regression** | frames t-4..t | scalar speed | Motion magnitude |
| **Segmentation** | frame t | foreground mask | Foreground/background separation |
| **Rotation Contrast** | two rotated versions | same rotation? | Rotation-invariant features |

A **single reservoir forward pass** through a sequence produces states used by **all five heads** simultaneously.

## Multi-Task Loss Balancing

Loss weights λ are chosen so each task contributes roughly equally:
```
L_total = λ_fp·MSE(pred_frame, true_frame)
        + λ_to·BCE(order_pred, order_label)
        + λ_speed·MSE(speed_pred, speed_true)
        + λ_seg·BCE(mask_pred, mask_true)
        + λ_rc·BCE(rot_same, rot_label)
```

Initial λ set heuristically (e.g., all = 1.0), then tuned on validation set to prevent one task from dominating.

## Does MTL Help?

**Hypothesis**: Joint training yields **better general features** than any single pretext task alone, because:
- No task can "cheat" using shortcuts that don't generalize
- Tasks regularize each other (multi-task regularization)
- Reservoir learns comprehensive dynamics vocabulary

**Expected ablation results** (ordered by downstream accuracy):
1. Full 5-task pre-training (best)
2. Frame + Temporal Order + Speed (no segmentation/rotation)
3. Frame prediction only (worst)

## Closed-Form Multi-Task

Unlike deep multi-task (where gradients conflict), CluSTAR's ridge-regression multi-task has **no gradient interference** — each task solved separately on same X, then concatenated. This means MTL here doesn't have the typical negative transfer seen in deep networks; all tasks are genuinely complementary.

## Connection to SSL Literature

CluSTAR's 5-task suite resembles:
- **VideoSSL** (predicting, ordering, speed) — direct inspiration
- **Shuffle & Learn** (temporal order) — same idea
- **Rotation Prediction** (image-based SSL) → adapted to rotation contrast (pairwise)
- **Masked Autoencoding** (frame prediction with masking) — could extend CluSTAR

Multi-task self-supervision is a **general recipe**: pick diverse free supervisory signals, solve jointly, get robust features.
