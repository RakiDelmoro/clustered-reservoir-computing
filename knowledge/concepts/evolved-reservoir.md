---
title: "Evolved Reservoir"
aliases: [Neuroevolution-Optimized Reservoir, Evolvable Reservoir]
tags: [reservoir-computing, neuroevolution, architecture-search, optimization]
sources:
  - "daily/2026-04-16.md"  # Mentioned as reservoir variant
created: 2026-04-16
updated: 2026-04-16
word_count: 220
---

# Evolved Reservoir

An **Evolved Reservoir** uses **neuroevolution** (evolutionary algorithms) to optimize reservoir connectivity patterns, spectral radii, or neuron properties rather than using purely random weights.

## Motivation

Random ESN reservoirs may be suboptimal for specific tasks. Evolutionary search can discover structured connectivity (e.g., small-world, modular) that outperforms Erdős–Rényi random graphs while maintaining computational efficiency.

## What Can Be Evolved?

1. **Weight connectivity pattern** (binary adjacency matrix) — structure, not magnitudes
2. **Reservoir topology** — cluster assignments, modularity, small-world parameters
3. **Neuron parameters** — leaking rates (α), activation thresholds
4. **Input-to-reservoir routing** — which input features connect to which neurons
5. **Multi-reservoir architecture** — number of layers, inter-layer connections

## Evolutionary Algorithms Used

- **Genetic Algorithms** (GA) — binary genomes encoding adjacency patterns
- **NEAT** (NeuroEvolution of Augmenting Topologies) — grows structure over generations
- **CMA-ES** (Covariance Matrix Adaptation) — continuous optimization of real-valued parameters
- **Particle Swarm** — global optimization with less computational overhead

## Fitness Function

Typically measured by:
- Validation accuracy on target task (e.g., action classification accuracy)
- Memory capacity (linear memory + nonlinear memory)
- Robustness to noise
- Multi-objective: accuracy + sparsity + spectral properties

## Challenges

- **Computationally expensive** — evaluate fitness across many generations
- **Stochastic evaluations** — reservoir randomness adds noise to fitness estimates
- **Overfitting risk** — evolved structure may overfit validation set
- **Reproducibility** — evolutionary runs are stochastic; multiple seeds needed

## Role in CluSTAR

CluSTAR's **clustered, multi-timescale architecture** was inspired by evolved reservoir findings (small-world topology, heterogeneous dynamics) but hand-designed based on domain knowledge for moving MNIST. An interesting future extension: use neuroevolution to optimize CluSTAR's clustering assignments, routing patterns, or per-cluster spectral radii for specific embedded hardware constraints.
