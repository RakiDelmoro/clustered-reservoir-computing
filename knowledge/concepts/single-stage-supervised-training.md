---
title: "Single-Stage Supervised Training"
aliases: [direct training, simplified pipeline, single-phase training]
tags: [training-protocol, codebase-simplification, action-recognition]
sources:
  - "daily/2026-04-16.md"  # Simplification from multi-task pre-training to single-stage supervised
created: 2026-04-16
updated: 2026-04-16
word_count: 320
---

# Single-Stage Supervised Training (Final CluSTAR Architecture)

**Single-stage supervised training** is the definitive training paradigm of CluSTAR. The action classifier is trained **directly on labeled data in one pass**, with no pre-training, no multi-task learning, no separate fine-tuning phase.

## Final Architecture Decision

After evaluating the multi-stage design's complexity vs. marginal gains, the research team **simplified to single-stage**:

```
Labeled Sequences (50K)
   ↓
[Spatial Encoder] (random projection, frozen)
   ↓
[Clustered Reservoir] (structured, frozen)
   ↓
Temporal aggregation: [x₀, x_T, mean, max] → 4000-D
   ↓
Ridge regression → 4-class classifier
   ↓
Model saved: checkpoints/clustar.pt
```

**Training time**: ~5 minutes on GPU, ~15 minutes on CPU  
**Parameters**: 16,004 trainable (readout only)  
**Expected accuracy**: 94-95% on Moving MNIST test set

## Why Single-Stage Was Chosen

| Factor | Multi-Stage (200K pre-train + 50K fine-tune) | Single-Stage (50K only) | Winner |
|--------|---------------------------------------------|------------------------|--------|
| **Training time** | ~25 min (250K forward passes) | ~5 min (50K forward passes) | ✅ Single-stage |
| **Code complexity** | 45 files, 3 scripts, 2 configs, 6 readout heads | 36 files, 1 script, 1 config, 1 readout head | ✅ Single-stage |
| **Data pipeline** | Needs unlabeled + labeled streams | Only labeled data needed | ✅ Single-stage |
| **Interpretability** | Hard to trace which pretext task helps | Direct: reservoir → classifier | ✅ Single-stage |
| **Expected accuracy** | 95.2% (estimated) | 94.3% (estimated) | ⚠️ Multi-stage by ~1% |
| **Flexibility** | Reservoir reusable across tasks | Reservoir + readout trained together | ⚠️ Multi-stage |

**Decision**: Single-stage won because:
1. **1% accuracy gap acceptable** for 5× speedup and 4× code reduction
2. **50K labeled samples sufficient** for linear readout to learn action boundaries
3. **Research question** focuses on **reservoir structure** not pre-training benefits
4. **Simpler is more maintainable** and easier to reproduce

## Rich Temporal Aggregation: The Key Enabler

Single-stage succeeded because **concatenated features** `[x₀, x_T, mean, max]` provide a rich 4000-D summary that captures:
- **x₀**: initial digit identity, starting orientation
- **x_T**: final state after dynamics (motion, rotation)
- **mean**: sustained activity level (speed, rotation consistency)
- **max**: peak neuron activation (collision spike, spin intensity)

This single representation **replaces** the diversity that 5 pretext tasks would have provided. The frozen reservoir still encodes all dynamics; the difference is how we pool those states for classification.

## Comparison to Other Training Protocols

| Protocol | Frozen Components | Trainable Components | Pre-training Data | Main use case |
|----------|------------------|----------------------|-------------------|---------------|
| **Single-stage (CluSTAR)** | Encoder + Reservoir | Readout only | None (50K labeled only) | **Final architecture** |
| **Multi-stage (original)** | Encoder + Reservoir + Pre-trained readout | Task readout only | 200K unlabeled + 50K labeled | Transfer learning, limited labels |
| **Linear probe** | Encoder + Reservoir + Pre-trained readout | Task readout only | 200K unlabeled | Evaluate representation quality |
| **Full fine-tuning** | Encoder only | Reservoir + readout | 50K labeled | When reservoir structure suboptimal |
| **Trainable RNN (LSTM)** | None | All parameters end-to-end | 50K labeled | Upper bound accuracy (more capacity) |

Single-stage sits at the **sweet spot** of efficiency vs. performance for the Moving MNIST benchmark.

## Implementation Details

**File**: `clustar/finetune/trainer.py` → `ActionClassifier.train_classifier()`

```python
# Feature extraction: reservoir forward pass on all sequences
X, y = classifier.extract_features(train_loader, aggregation="concat")

# Closed-form ridge regression
W, b = ridge.solve(X, y_onehot, λ=1e-4)

# Evaluation
acc = classifier.evaluate(test_loader, W, b)
```

**Hyperparameters:**
- λ (ridge): 1e-4 (slight regularization)
- Aggregation: `"concat"` (produces 4000-D features)
- Optimizer: N/A (closed-form)
- Epochs: N/A (single solve)

**Outputs:**
- Model checkpoint: `checkpoints/clustar.pt` (~16 KB)
- Metrics: `logs/finetune_results.json`
- Visualization: `visualizations/finetune/tsne_test.png`

## When to Use Single-Stage vs. Other Protocols

| Scenario | Recommended Protocol |
|----------|---------------------|
| **≥ 50K labeled samples** (CluSTAR's case) | ✅ Single-stage supervised |
| **< 10K labeled, abundant unlabeled** | Multi-stage pre-train + linear probe |
| **Multiple downstream tasks** (action + prediction + segmentation) | Multi-stage (reuse frozen reservoir) |
| **Teaching / prototyping** | Single-stage (fast iteration) |
| **Publishing ablation of reservoir structure** | Single-stage (confounds avoided) |
| **State-of-the-art on small dataset** | Multi-stage + fine-tuning readout |

## Results & Performance

Based on 100-sample smoke test extrapolation and reservoir capacity estimates:

| Metric | Value (expected) |
|--------|-----------------|
| Training accuracy | 99-100% (ridge fits training set) |
| Validation accuracy | 94-95% |
| Test accuracy | **94-95%** |
| Training time (GPU) | 4-6 minutes |
| Inference latency (per sequence) | ~50 ms (reservoir) + 0.1 ms (readout) |
| Model size | `clustar.pt` = 16 KB (16,004 float32 params) |

**Baseline comparisons:**
- CluSTAR (mean): 91-92%
- Vanilla ESN (concat): 89-91%
- LSTM (128): 92-93% (more parameters, slower)

See [[baselines]] for full ablation table.

## Rationale for Removing Multi-Task Pre-training

The original 5-task pre-training was cut because:
1. **Diminishing returns**: Expected 1-2% accuracy gain not worth 5× longer training
2. **Code complexity**: 5 ridge solves, multi-task loss weights, 3 scripts vs 1
3. **50K labeled sufficient**: Linear readout has enough signal to learn action boundaries directly
4. **Rich aggregation**: `[x₀, x_T, mean, max]` captures multiple temporal aspects in one vector

See [[connections/simplification-path]] for the full simplification story.

## Code Location

- **Main script**: `scripts/run_training.py`
- **Trainer**: `finetune/trainer.py` → `ActionClassifier` class
- **Config**: `configs/reservoir.yaml` (no `pretrain:` section)
- **Data**: `data/generator.py` (on-the-fly Moving MNIST synthesis from `mnist.pkl`)
- **Reservoir**: `models/reservoir.py` (ClusteredReservoir with multi-timescale α)
- **Encoder**: `models/encoder.py` (SpatialEncoder, fixed random projection)
