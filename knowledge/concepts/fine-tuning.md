---
title: "Fine-tuning"
aliases: [readout fine-tuning, adaptive readout]
tags: [training-protocol, transfer-learning, adaptation]
sources:
  - "daily/2026-04-16.md"  # Fine-tuning readout decision
created: 2026-04-16
updated: 2026-04-16
word_count: 180
---

# Fine-tuning

**Fine-tuning** in CluSTAR context means **adapting the pre-trained readout layer** to the downstream task while keeping the reservoir and spatial encoder frozen.

## CluSTAR Fine-tuning Protocol

After self-supervised pre-training (200K unlabeled sequences):
1. **Freeze**: reservoir weights (W_res, W_in), spatial encoder (R), pre-trained readout initialization
2. **Collect**: reservoir states on 50K labeled sequences
3. **Train**: new task-specific readout `W_action` (or slightly adjust `W_readout` weights) via ridge regression with small λ (e.g., 1e-4)

```
W_action = argmin ||Y_action - W_action·X_frozen||² + λ||W_action||²
```

## Linear Probe vs Fine-tuning

| | Linear Probe | Fine-tuning |
|---|--------------|-------------|
| What adapts? | Only new head | Pre-trained readout + new head |
| Regularization | Strong (frozen) | Weaker (small λ allows drift) |
| Expected accuracy | Slightly lower | Slightly higher |
| What it measures | Representation quality | Representation + readout adaptability |
| Risk | None | Catastrophic forgetting of pre-trained features |

## Why Fine-tuning Chosen for CluSTAR

With **50K labeled examples**, there is enough data to slightly adjust pre-trained readout weights to better align with action classification targets without destroying general dynamics features. The small ridge penalty prevents overfitting.

## Alternatives

- **Full model fine-tuning** (end-to-end with small LR) — not feasible; reservoir not differentiable; would need backprop through reservoir which defeats RC purpose
- **Adapter modules** — insert small trainable layers between reservoir and readout — possible extension
- **No fine-tuning** (pure linear probe) — simpler baseline

## Expected Outcome

Fine-tuning should yield **1–3% higher accuracy** than linear probe, confirming pre-trained features are good but not perfectly aligned with action labels. Small gap indicates representation quality is high.
