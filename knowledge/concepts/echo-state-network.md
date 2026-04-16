---
title: "Echo State Network"
aliases: [ESN]
tags: [reservoir-computing, recurrent-neural-network, temporal-dynamics]
sources:
  - "daily/2026-04-16.md"  # Discussion of ESN as baseline
created: 2026-04-16
updated: 2026-04-16
word_count: 280
---

# Echo State Network (ESN)

An **Echo State Network** is the canonical form of reservoir computing, introduced by **Jaeger (2001)**. ESNs use a fixed, sparse, randomly initialized recurrent reservoir with an echo state property that ensures fading memory of past inputs.

## Architecture

```
Input u(t) → W_in (random) → Reservoir State x(t) → W_out (trained) → Output y(t)
```

- **Reservoir state update**: x(t+1) = tanh(W_in·u(t+1) + W_res·x(t))
- **W_res**: sparse (~10% connectivity), spectral radius ρ < 1.0
- **W_out**: trained via ridge regression (closed-form solution)

## Key Hyperparameters

| Parameter | Typical Range | Role |
|-----------|--------------|------|
| Reservoir size N | 500–2000 neurons | Capacity vs. compute |
| Spectral radius ρ | 0.7–0.99 | Memory depth (higher = longer memory) |
| Input scaling σ_in | 0.1–0.5 | Nonlinear response regime |
| Connectivity sparsity | 1–10% | Sparsity vs. richness |

## Memory Capacity

The **memory capacity** of an ESN quantifies how many past inputs can be reconstructed from the current state. For a reservoir of size N, maximum memory capacity ≈ N. Memory decays exponentially with lag; spectral radius controls decay rate.

## Training

Only output weights are trained:
```
W_out = Y_train · X_train^T · (X_train · X_train^T + λI)^{-1}
```
where λ is L2 regularization. No gradient descent through reservoir.

## Limitations

- Random connectivity may be suboptimal for structured tasks (e.g., vision)
- Single timescale for all neurons
- No explicit spatial feature routing
- Manual hyperparameter tuning required

## Role in CluSTAR

ESN serves as the **baseline** against which CluSTAR's structured innovations (clustering, multi-timescale, input routing) are compared. The vanilla ESN implementation (`models/vanilla_esn.py`) uses the same spatial encoder but random unstructured reservoir.
