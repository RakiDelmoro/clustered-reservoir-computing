---
title: "Multi-Timescale Dynamics"
aliases: [heterogeneous timescales, leaking rates, fading memory hierarchy]
tags: [temporal-dynamics, reservoir-computing, neuron-model]
sources:
  - "daily/2026-04-16.md"  # Multi-timescale neuron design
created: 2026-04-16
updated: 2026-04-16
word_count: 240
---

# Multi-Timescale Dynamics

**Multi-timescale dynamics** refers to incorporating neurons with different **leaking rates** (memory decay constants) within the same reservoir. Fast neurons react quickly to inputs; slow neurons retain information longer.

## Leaky Integration in ESNs

Standard ESN update with leaking rate α ∈ (0,1]:
```
x(t+1) = (1-α)·x(t) + α·tanh(W_in·u(t+1) + W_res·x(t))
```

- **α = 1.0** → no leak (pure tanh recurrence)
- **α → 0** → very slow integration (long memory)
- **α → 1** → fast integration (short memory)

## Heterogeneous α Values in CluSTAR

Three neuron groups per cluster (or per cluster type):

| Group | Leaking α | Memory Horizon | Role |
|-------|-----------|----------------|------|
| Fast | α ≈ 0.3 | ~5 timesteps | High-frequency motion (vibration, quick digit bounces) |
| Medium | α ≈ 0.6 | ~15 timesteps | Smooth trajectories, medium-term trends |
| Slow | α ≈ 0.9 | ~50+ timesteps | Identity, long-term context (which digit is it?) |

**Effect**: Single reservoir simultaneously represents dynamics at multiple timescales without stacking layers.

## Why Multi-Timescale Matters

1. **Natural world has multi-scale dynamics** — fast jitter vs slow drift
2. **Single α is a bottleneck** — cannot capture both fast bouncing and slow spinning well
3. **Heterogeneous neurons increase capacity** without increasing reservoir size

## Theoretical Connection

This relates to:
- **Multiple Timescales RNN (MTRNN)** (Yamashita & Tani, 2008) — explicit hierarchy of timescales for developmental learning
- **Hierarchical RNNs** — layered RNNs with different clock rates
- **Liquid State Machine** — LIF neurons have different membrane time constants

In reservoir computing, leaking rate is set **per neuron or neuron group** (not learned). CluSTAR sets it by cluster function.

## Implementation Detail

In PyTorch, leaky ESN update:
```python
x_prev = ...  # [batch, N]
x_new = (1 - alpha) * x_prev + alpha * torch.tanh(W_in @ u + W_res @ x_prev)
```
Vector `alpha` can be per-neuron (N,) or per-group broadcast.

## Ablation Value

Expected results:
- **Uniform α (all medium)** → degraded performance on fast bouncing (blurring) and slow spinning (forgetting)
- **Multi-α** → each timescale component captured optimally
