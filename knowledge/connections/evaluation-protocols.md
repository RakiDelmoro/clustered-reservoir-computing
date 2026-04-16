---
title: "Evaluation Protocols: Linear Probe vs. Fine-tuning"
source: "daily/2026-04-16.md"
tags: [evaluation, representation-learning, transfer-learning]
---

# Linear Probe vs. Fine-tuning: Diagnostic Tools

Two standard protocols to evaluate self-supervised representations: **linear probe** (train only linear classifier) and **fine-tuning** (adjust pre-trained readout). Both diagnose different aspects of learned features.

## Definitions in CluSTAR Context

**Linear Probe**
- Freeze: spatial encoder + reservoir + pre-trained readout
- Collect: frozen reservoir states from labeled sequences
- Train: `W_action ∈ ℝ^(4×1000)` via ridge regression (no modification to pre-trained weights)
- Test: end-to-end accuracy

**Fine-tuning** (of readout)
- Freeze: spatial encoder + reservoir only
- Train: `W_action` with ridge regression but **initialize** it from pre-trained readout's action head (if exists) or from zero
- Optionally: small λ allows weights to deviate from pre-trained initialization

## What Each Measures

| Protocol | Measures | High Score Means |
|----------|----------|------------------|
| **Linear Probe** | **Representation quality** directly | Pre-trained dynamics features are linearly separable for actions |
| **Fine-tuning** | **Representation + readout adaptability** | Either features are good OR readout can shift to align with labels |

## Expected Gap

- **Gap = Fine-tune acc − Linear Probe acc**
- Small gap (0–2%) → features already well-adapted, just need slight scaling
- Large gap (5%+) → features need non-linear transformation or readout re-alignment

CluSTAR expects **small gap** because pre-training includes an action classification head during multi-task learning (though it's trained on 200K unlabeled with noisy/missing labels). If labels were present in pre-training, gap should be near zero.

## Why Not Full Fine-tuning?

Full fine-tuning (adjusting reservoir + encoder) would require **backpropagation through the reservoir**, which:
- Defeats purpose of reservoir computing (fixed weights)
- Loses embedded efficiency (no BPTT on microcontroller)
- Introduces gradient vanishing/exploding in reservoir

Thus only **readout fine-tuning** is meaningful for RC.

## Literature Standard

- **Supervised pre-training + linear probe** (ImageNet features → logistic regression on new dataset)
- **Self-supervised pre-training + linear probe** (MoCo, SimCLR, MAE all report this)
- **Full fine-tuning** reported as supplementary

CluSTAR follows SSL convention: **linear probe primary**, fine-tune readout secondary.

## Decision for CluSTAR

**Final protocol** (from conversation):
- **Primary metric**: Fine-tune readout (λ=1e-4, initialized from pre-trained readout)
- **Secondary**: Linear probe (as ablation of readout adaptation)
- **Compare**: Both vs. Random Reservoir (no pre-train) fine-tune

Rationale: With 50K labeled examples, readout has enough data to slightly adapt; fine-tuning captures slight improvement over frozen features. But also report linear probe for completeness.
