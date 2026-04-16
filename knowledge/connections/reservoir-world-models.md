---
title: "Reservoirs as World Model Backbones"
source: "daily/2026-04-16.md"
tags: [world-model, reservoir-computing, robotics, embedded-ai]
---

# Reservoir Computing for World Models

Using a **reservoir** as the dynamics backbone of a world model combines the predictive power of recurrent networks with the **training efficiency** of reservoir computing.

## Problem: Traditional RNNs as World Models

World models (MuZero, Dreamer) use **trained RNNs** (LSTM/GRU) to predict environment dynamics. Issues:
- **BPTT** required → high memory, slow training
- **Many parameters** → data-hungry, poor sample efficiency
- **Training instability** — gradients vanish/explode over long horizons
- **Embedded inference** — RNN weights large, need full backprop for adaptation

## Reservoir Solution

A **fixed recurrent reservoir** replaces the trained RNN:

```
State update: x(t+1) = tanh(W_in·u(t) + W_res·x(t))   ← Fixed
Readout:       ŷ(t+1) = W_out·x(t)                    ← Trained
```

Only W_out is trained; W_in, W_res random and frozen.

**Advantages for world models**:
- **No BPTT through reservoir** → faster training, no memory bottleneck
- **Online adaptation** → W_out can update online as new data arrives (continual learning)
- **Small parameter count** → efficient on-device; W_out is linear
- **Embedded-friendly** → microcontroller can run reservoir forward with O(N²) multiplies

## CluSTAR's World Model Head

CluSTAR extends vanilla reservoir world model with:
- **Spatial encoder** → compact visual features
- **Structured reservoir** → better inductive bias for visual dynamics
- **Multi-task pre-training** → rich dynamics features before world model fine-tuning

World model head predicts **next frame pixels** (or k-step ahead). Loss: MSE(pred_frame, actual_frame).

## Sample Efficiency

Because readout training is **closed-form ridge regression**, world model can learn from **few examples**:
- Vanilla ESN: 100 trajectories may suffice for simple dynamics
- CluSTAR: structure reduces sample need further → 50 trajectories may work

Contrast with LSTM world model: needs thousands of trajectories for end-to-end SGD.

## Limitations

- **Fixed capacity** — reservoir size N caps expressivity; cannot increase capacity by training
- **No internal adaptation** — reservoir cannot learn new patterns without changing W_out
- **Suboptimal for complex dynamics** — random connectivity may miss task-specific structure
- CluSTAR mitigates with structured design, but still not as flexible as fully-trained RNN

## When to Choose Reservoir World Model

**Use reservoir world model when:**
- Embedded/robotics: microcontroller-level compute
- Data scarce: need sample-efficient learning
- Online adaptation required: readout updates on-the-fly
- Dynamics are relatively simple (bouncing digits, pendulum, simple robot)

**Use trained RNN world model when:**
- Complex, high-dimensional dynamics (3D scenes, cloth simulation)
- Large offline dataset available
- No strict compute constraints (server/GPU)
- Need maximum accuracy

## Research Question

CluSTAR investigates: *Can structured reservoir computing match trained RNNs on visual world modeling while using <1% of the parameters and enabling online learning?* Moving MNIST is the first step; future work: real robotic manipulation datasets.
