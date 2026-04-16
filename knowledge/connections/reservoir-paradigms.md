---
title: "Reservoir Computing Paradigms"
source: "daily/2026-04-16.md"
tags: [reservoir-computing, taxonomy]
---

# Reservoir Computing: A Taxonomy of Paradigms

**Reservoir computing** encompasses several architectural families distinguished by neuron models, connectivity, and training approaches. These paradigms share the **fixed recurrent + trainable readout** principle but differ in implementation.

## Hierarchy

```
Reservoir Computing
├── Echo State Network (ESN)
│   └── Deep ESN (stacked)
├── Liquid State Machine (LSM)
├── Evolved Reservoir (neuroevolution-optimized)
└── CluSTAR (structured ESN variant)
```

## Key Distinctions

### Neuron Model
- **ESN / Deep ESN**: continuous sigmoidal (tanh)
- **LSM**: spiking integrate-and-fire
- **Evolved / CluSTAR**: can use either; CluSTAR uses tanh

### Connectivity
- **ESN**: sparse random Erdős–Rényi graph
- **Deep ESN**: multiple random reservoirs + inter-layer weights
- **Evolved Reservoir**: structured but *evolved* (not hand-designed)
- **CluSTAR**: structured (clusters + small-world) *by design*

### Timescales
- **ESN**: homogeneous (single α per neuron, usually random)
- **LSM**: heterogeneous (LIF neurons have diverse τ_m)
- **Deep ESN**: hierarchical timescales per layer
- **Evolved Reservoir**: can evolve per-neuron α
- **CluSTAR**: heterogeneous *by cluster* (3 groups per cluster)

### Input Representation
- **ESN / Deep ESN**: continuous vector
- **LSM**: spike trains (often Poisson)
- **CluSTAR**: routed feature streams after spatial encoder

## When to Use Which?

| Paradigm | Best For | Hardware | Sample Efficiency |
|----------|----------|----------|-------------------|
| **Vanilla ESN** | Quick prototyping, generic time-series | CPU-friendly | Medium |
| **LSM** | Neuromorphic hardware, spike-based sensors | Loihi, SpiNNaker | High (event-driven) |
| **Deep ESN** | Long sequences needing hierarchy | GPU/CPU | Low (more params) |
| **Evolved Reservoir** | Task-specific optimization; offline design | Any (after evolution) | Very High (specialized) |
| **CluSTAR** | Visual dynamics with semantic structure; embedded | Microcontroller | High (structured bias) |

## Research Trajectory

The field moved from **random** (ESN) → **spiking** (LSM) → **deep** (Deep ESN) → **optimized** (Evolved) → **structured** (CluSTAR). CluSTAR can be seen as a **hand-coded expert architecture** that mimics what neuroevolution might discover for moving digit tasks.
