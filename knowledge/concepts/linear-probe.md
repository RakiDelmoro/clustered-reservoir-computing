---
title: "Linear Probe"
aliases: [linear evaluation, frozen representation test]
tags: [evaluation-protocol, representation-learning, self-supervised-learning]
sources:
  - "daily/2026-04-16.md"  # Linear probe vs fine-tuning discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 200
---

# Linear Probe

A **linear probe** evaluates the quality of a pre-trained representation by training a **linear classifier** on frozen features. It tests whether the learned representations are linearly separable for the downstream task.

## Procedure

1. Pre-train model (reservoir + readout) on self-supervised pretext tasks using unlabeled data
2. Freeze **all** pre-trained weights (reservoir, encoder, pre-trained readout)
3. Collect reservoir states from labeled downstream dataset
4. Train **only** a linear classifier (no hidden layers) on these features
5. Evaluate on test set

**Classifier**: typically W ∈ ℝ^(C×N) where C = number of downstream classes, N = feature dimension; trained via logistic regression or ridge classification.

## Purpose

Linear probe **isolates representation quality** from classifier capacity:
- If frozen features yield high accuracy → pre-training learned good features
- If fine-tuning needed → features are suboptimal or task mismatch

Standard in self-supervised learning (SSL) literature:
- **SimCLR**, **MoCo**, **MAE** all report linear probe accuracy on ImageNet
- Linear probe is a **lower bound** on what's achievable with full fine-tuning

## CluSTAR Protocol

In CluSTAR pre-training:
- Pre-trained features: frozen reservoir states from pre-training phase
- Linear probe: ridge regression action classifier (no adaptation of pre-trained readout)
- Compared against **fine-tuning readout** (slight adjustment of pre-trained readout weights)

**Expected**: Multi-task pre-training should yield features where action classes are nearly linearly separable, so linear probe accuracy should be close to fine-tuning accuracy.

## Advantages

- Clean, interpretable comparison across pre-training variants
- Fast evaluation (no backprop through large model)
- Reduces confounding from overparameterized classifiers

## Limitations

- Underestimates true representation utility if task requires non-linear combination of features
- Not always correlated with full fine-tuning performance (some SSL methods show gaps)
