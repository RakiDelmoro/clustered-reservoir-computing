---
title: "Baselines"
aliases: [comparison methods, ablation studies]
tags: [evaluation, experimental-design, ablation]
sources:
  - "daily/2026-04-16.md"  # Baselines discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 300
---

# Baselines and Ablation Methods

Comprehensive evaluation of CluSTAR requires comparing against several **baseline models** and **ablated variants** to isolate the contribution of each architectural innovation.

## Model Baselines

| Baseline | Description | What It Tests |
|----------|-------------|---------------|
| **Vanilla ESN** | Unstructured random reservoir (Erdős–Rényi), single timescale, flat input, no routing | Does structure improve over random? |
| **1-Layer LSTM** | 128-unit LSTM trained end-to-end with Adam | Does RC approach compete with gradient-based RNNs? |
| **Random Reservoir + Data Augmentation** | CluSTAR structure but trained only on labeled data with augmentations (rotations, translations, noise) | Can data augmentation replace pre-training? |
| **Supervised from Scratch** | CluSTAR (random init) trained directly on 50K labeled with multi-task loss | Does pre-training provide initialization benefit? |
| **CluSTAR (No Pre-train)** | Structured reservoir + spatial encoder, readout trained directly on action labels (no self-supervised phase) | Ablates pre-training value; tests if structure alone suffices |

## Architecture Ablations (within CluSTAR)

To measure contribution of each component:

| Ablation | Removed Component | Expected Effect |
|----------|------------------|----------------|
| **No Clustering** | Replace clustered W_res with fully random connectivity | Loss of functional specialization; performance drop |
| **Uniform α** | All neurons have same leaking rate (α=0.6) | Cannot capture multi-timescale dynamics; accuracy drop on fast + slow actions |
| **No Input Routing** | All features connect to all clusters | Increased interference; reduced interpretability; possible accuracy drop |
| **No Spatial Encoder** | Raw pixels → reservoir directly | High-dimensional input dilutes dynamics; slower convergence |
| **Single Timescale × Cluster** | Only one α per cluster instead of 3× neurons | Coarser temporal resolution; some actions harder to classify |

## Pretext Task Ablations

To measure which self-supervised objectives matter:

| Ablation | Removed Tasks | Hypothesis |
|----------|---------------|------------|
| **Only Frame Prediction** | Remove all but frame_pred | captures dynamics but loses temporal/semantic structure |
| **No Temporal Order** | Remove temporal_order | model loses arrow-of-time bias; may confuse forward/backward |
| **No Speed Regression** | Remove speed_regression | motion magnitude understanding degraded |
| **No Segmentation** | Remove segmentation | no explicit objectness; may attend to background |
| **No Rotation Contrast** | Remove rotation_contrast | rotation invariance lost; spinning detection harder |

Typically evaluated by re-running pre-training with subset of tasks, then measuring downstream action accuracy.

## Evaluation Metrics Compared

All baselines measured on:
- **Primary**: Action classification accuracy (test set)
- **Secondary**: Frame prediction MSE (world model quality)
- **Efficiency**: Parameter count, inference latency (estimated), RAM usage
- **Ablation-specific**: Memory capacity, separation property (ESN metrics)

## Statistical Significance

With 10K test sequences, standard deviation of accuracy estimate ≈ √(p(1-p)/10000) ≈ 0.5% for p=50%. All reported differences >2% are statistically significant (p < 0.05).

## Expected CluSTAR Advantages

| Metric | Vanilla ESN | LSTM | CluSTAR |
|--------|-------------|------|---------|
| Accuracy | ~65% baseline | ~70% | **~78%** target |
| Parameters | ~10K readout | ~100K total | ~10K readout |
| Inference speed | Very fast | Moderate | Very fast |
| Online adaptation | Yes (ridge) | No (needs SGD) | Yes |
| Interpretability | Low | Low | **High** (clusters) |
