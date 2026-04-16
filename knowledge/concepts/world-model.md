---
title: "World Model"
aliases: [predictive dynamics model, environment model]
tags: [reinforcement-learning, robotics, dynamics-modeling, planning]
sources:
  - "daily/2026-04-16.md"  # World model backbone discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 300
---

# World Model

A **world model** is a predictive model of an environment's dynamics that enables an agent to plan actions by simulating future states. World models learn the **transition function** P(s'|s,a) or its deterministic counterpart s' = f(s,a).

## Role in RL & Robotics

World models allow:
- **Model-based planning** (MPC, MCTS) using imagined rollouts
- **Data augmentation** — generate synthetic transitions
- **Feature learning** — learn reusable state representations
- **Safety** — test policies in simulation before real-world execution

## Common Architectures

### 1. **Deterministic Models**
```
s' = f_θ(s, a)
```
- Feedforward or recurrent networks
- Used in **MuZero** (combined with value/policy heads)

### 2. **Stochastic Models**
```
s' ~ P_θ(s'|s, a)
```
- Variational Autoencoders (VAEs) + transition prior
- Used in **Dreamer**, **PlaNet**

### 3. **Hybrid Models**
```
s' = deterministic_part + stochastic_noise
```
- RSSM (Recurrent State-Space Model) in Dreamer v2

## Reservoir Computing as World Model Backbone

**Traditional RNNs** struggle with:
- Gradient vanishing/exploding in long sequences
- Slow BPTT training
- High parameter count → poor sample efficiency

**Reservoir Computing** (ESN/CluSTAR) offers:
- **No BPTT** — fixed reservoir avoids gradients
- **Fast training** — linear readout via ridge regression
- **Online adaptation** — readout updates incrementally
- **Embedded-friendly** — low compute, memory-efficient

**Trade-off**: Less flexible than trained RNNs; may underfit complex dynamics unless structured (as in CluSTAR).

## CluSTAR as World Model

In our architecture:
- **Reservoir** = temporal dynamics encoder (fixed)
- **Frame prediction head** = world model predictor (trained readout)
- Given frames I₁...Iₜ → reservoir states x(t) → predict I_{t+1}, I_{t+2}, ...

**Advantages for embedded robotics**:
- Microcontroller can run reservoir forward pass cheaply
- World model updates online as new observations arrive
- No backpropagation needed on-device

## Evaluation Metrics

- **Prediction MSE** between predicted and actual future frames
- **Multi-step roll-out quality** (does error accumulate?)
- **Planning performance** (if used in MPC: task success rate)
- **Sample efficiency** (how many real trajectories needed?)
