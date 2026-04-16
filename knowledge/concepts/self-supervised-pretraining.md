---
title: "Self-Supervised Pre-training"
aliases: [unsupervised pre-training, pretext tasks, multi-task self-supervision]
tags: [self-supervised-learning, multi-task-learning, representation-learning, reservoir-computing]
sources:
  - "daily/2026-04-16.md"  # Pre-training design discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 380
---

# Self-Supervised Pre-training for Reservoir World Models

**Self-supervised pre-training** for reservoir computing involves training the readout layer(s) on **pretext tasks** using large amounts of **unlabeled sequential data**, then freezing the reservoir and readout for downstream tasks.

## Why Pre-train Reservoir Readouts?

In reservoir computing, the reservoir weights are fixed random. "Pre-training" does **not** update reservoir weights (there are no gradients). Instead:

1. Run reservoir on unlabeled sequences → collect reservoir states X
2. Train readout weights W via ridge regression to solve one or more pretext tasks
3. Freeze reservoir + pre-trained readout
4. For downstream task (e.g., action classification), either:
   - **Linear probe**: use frozen pre-trained features + new linear head
   - **Fine-tune readout**: slightly adjust pre-trained readout weights on labeled data

This approach leverages unlabeled data to learn **general-purpose temporal features** before seeing labeled examples.

## Multi-Task Pretext Tasks in CluSTAR

Five joint objectives (weighted sum loss):

| Task | Desired Feature | Target | Loss |
|------|----------------|--------|------|
| **Frame Prediction** | Dynamics modeling | Next-frame pixels | MSE |
| **Temporal Order** | Arrow of time | Which window came first? | Binary cross-entropy |
| **Speed Regression** | Motion magnitude | Scalar velocity | MSE |
| **Segmentation** | Object-centric | Foreground mask | BCE |
| **Rotation Contrast** | Invariant representation | Same rotation? | Binary cross-entropy |

Multi-task training forces reservoir to encode a **rich, general-purpose dynamics representation** useful for multiple downstream applications.

## Training Procedure

```python
# Phase 1: Pre-training on unlabeled data (200K sequences)
X_states = reservoir.run(unlabeled_sequences)   # No gradients
W_readout = solve_ridge(X_states, pretext_targets)  # Closed-form

# Phase 2: Downstream fine-tuning (50K labeled)
W_action = ridge_regression(frozen_states, action_labels)
```

## Evaluation Protocols

**Linear Probe**: freeze everything, train only new linear classifier. Tests **representation quality** directly.

**Fine-tuning**: adjust pre-trained readout on downstream task. Tests if pre-training provides good initialization.

**Ablation**: remove pre-training (random reservoir) to measure pre-training value.

## Benefits

- **Data efficiency**: pre-trained features transfer to small labeled sets
- **Faster convergence**: downstream training requires fewer epochs
- **Robustness**: features generalize to unseen digit styles, velocities
- **Research contribution**: demonstrates unsupervised dynamics learning works

## Dataset Numbers (Moving MNIST)

- **Pre-training set**: 200,000 unlabeled sequences (no action labels)
- **Fine-tuning set**: 50,000 labeled sequences (with action labels)
- **Validation**: 10,000 labeled
- **Test**: 10,000 labeled

With 50K labeled, pre-training is optional but shows improved accuracy and faster convergence.

## Connection to Other SSL Methods

CluSTAR pre-training relates to:
- **SimCLR / MoCo** (contrastive learning) → rotation_contrast task
- **MAE** (masked autoencoding) → frame prediction with masking
- **DINO** (self-distillation) → could extend to teacher-student reservoirs
- **VideoSSL** (temporal order, speed prediction) → direct parallels

**Key difference**: RC pre-training uses **closed-form ridge regression**, not SGD; the reservoir itself is never updated via gradients.
