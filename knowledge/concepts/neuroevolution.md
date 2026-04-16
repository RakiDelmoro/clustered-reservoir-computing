---
title: "Neuroevolution"
aliases: [evolutionary algorithms, genetic algorithms, NEAT]
tags: [optimization, architecture-search, neuro-inspired-computing]
sources:
  - "daily/2026-04-16.md"  # Evolved reservoir and neuroevolution discussion
created: 2026-04-16
updated: 2026-04-16
word_count: 280
---

# Neuroevolution

**Neuroevolution** applies evolutionary algorithms (EAs) to optimize neural network architectures, weights, or hyperparameters. In reservoir computing, it can design better reservoir topologies than purely random initialization.

## Core Loop

```
1. Initialize population of genomes (encoding network structure/weights)
2. For each genome → decode → evaluate fitness (e.g., validation accuracy)
3. Select parents (tournament, roulette)
4. Apply variation: crossover + mutation
5. Replace population with offspring
6. Repeat for G generations
```

## Neuroevolution for Reservoirs

Can evolve:

1. **Connectivity patterns** — binary adjacency matrix (which neurons connect)
2. **Weight magnitudes** — continuous values subject to spectral radius constraint
3. **Neuron parameters** — leaking rates, activation thresholds
4. **Cluster assignments** — which neurons belong to which functional group
5. **Routing patterns** — input features → reservoir neuron mapping

**Genome encoding**: typically bitstring or real-valued vector representing edges or parameters.

## Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **GA (Genetic Algorithm)** | Bitstring genomes, crossover/mutation | Binary connectivity patterns |
| **NEAT** | Evolves topology + weights; speciation protects innovations | Growing reservoir structure |
| **CMA-ES** | Covariance Matrix Adaptation — continuous optimization | Per-neuron α, spectral radius tuning |
| **Particle Swarm** | Swarm intelligence, fewer hyperparams | Global optimization of W_res |

## Fitness Function

For reservoir design, fitness = validation performance on target task(s):
```
fitness = accuracy(action_classification) + w·memory_capacity
```
Multi-objective could also optimize for sparsity (fewer connections) or robustness.

## Advantages over Random Reservoirs

- **Task-specific structure**: evolved connectivity may align with input statistics
- **Discovering motifs**: small-world, modularity, rich-club may emerge naturally
- **Hardware-aware evolution**: can evolve for energy/latency constraints

## Disadvantages

- **Computationally expensive** — many reservoir evaluations per generation
- **Stochastic** — multiple runs needed; high variance
- **Overfitting risk** — may overfit validation set
- **Reproducibility** — random seeds matter

## Relation to CluSTAR

CluSTAR **does not use neuroevolution** — its clustered structure was hand-designed based on domain knowledge. An extension could evolve:
- Optimal number of clusters
- Inter-cluster connection probabilities
- Per-cluster spectral radii
- Input routing partition sizes

This would make CluSTAR **adaptive to hardware constraints** (e.g., few clusters for tiny microcontroller).

## Key References

- **Stanley & Miikkulainen (2002)** — NEAT algorithm
- **Yao (1999)** — Survey of evolutionary neural networks
- **Gallicchio & Micheli (2017)** — "Echo State Property of Deep Reservoirs" (includes evolution)
