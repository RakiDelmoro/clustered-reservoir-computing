# Knowledge Compilation Log

## 2026-04-16 — Session: Reservoir Computing World Model Research

**Session span**: 04:51 AM – ~1 hour  
**Messages**: 59  
**Topic**: Design of CluSTAR reservoir architecture for moving MNIST world model on embedded devices

### Concepts Extracted (19)

1. **reservoir-computing** — Fixed recurrent + trainable readout paradigm for temporal dynamics
2. **echo-state-network** — Classical ESN formulation, spectral radius, ridge training
3. **liquid-state-machine** — Spiking reservoir variant
4. **deep-echo-state-network** — Stacked reservoir layers
5. **evolved-reservoir** — Neuroevolution-optimized connectivity
6. **clustar** — Novel clustered, multi-timescale reservoir with input routing (our architecture)
7. **self-supervised-pretraining** — Multi-task pretext learning on unlabeled sequences
8. **world-model** — Predictive dynamics for planning; reservoir as efficient backbone
9. **moving-mnist** — Synthetic video benchmark with 4 action classes
10. **multi-task-learning** — Joint 5-task pre-training (frame, order, speed, seg, rotation)
11. **spatial-encoder** — Fixed random projection (784→128)
12. **clustered-reservoir** — 10×100 neurons, small-world connectivity, functional specialization
13. **multi-timescale-dynamics** — Heterogeneous leaking rates (α=0.3/0.6/0.9)
14. **input-routing** — Feature stream partitioning to dedicated clusters
15. **linear-probe** — Frozen representation evaluation
16. **fine-tuning** — Readout adaptation protocol
17. **baselines** — Vanilla ESN, LSTM, augmentation, no-pretrain ablations
18. **ridge-regression** — Closed-form readout training
19. **neuroevolution** — Evolutionary optimization (future extension)

### Connections Mapped (5)

1. **reservoir-paradigms** — Taxonomy of ESN/LSM/Deep/Evolved/CluSTAR
2. **clustar-components** — Data flow from encoder → routing → clustered reservoir → multi-timescale → readout
3. **multi-task-self-supervision** — Why 5 pretext tasks jointly yield richer dynamics features
4. **evaluation-protocols** — Linear probe vs fine-tuning: diagnostic differences
5. **reservoir-world-models** — Reservoir as efficient backbone vs traditional RNNs

### Key Decisions Recorded

- **Architecture name**: CluSTAR (Clustered Spatio-Temporal Reservoir)
- **Pre-training data**: 200K unlabeled sequences (no labels)
- **Fine-tuning data**: 50K labeled sequences (4 action classes)
- **Downstream protocol**: Fine-tune readout (primary), Linear probe (secondary)
- **Baselines**: Vanilla ESN, LSTM, Supervised-from-scratch, Data augmentation, CluSTAR w/o pre-train
- **Evaluation metrics**: Action accuracy (primary), frame MSE (secondary), efficiency metrics
- **Implementation**: All in `clustar/` Python package; on-the-fly sequence generation from `mnist.pkl`

### Outstanding Questions

- Optimal loss weights λ for multi-task pre-training (currently heuristic)
- Whether CluSTAR extends to real video (Kinetics, Something-Something)
- Neuroevolution of reservoir topology vs hand-designed clustering

### Files Created

```
knowledge/concepts/reservoir-computing.md
knowledge/concepts/echo-state-network.md
knowledge/concepts/liquid-state-machine.md
knowledge/concepts/deep-echo-state-network.md
knowledge/concepts/evolved-reservoir.md
knowledge/concepts/clustar.md
knowledge/concepts/self-supervised-pretraining.md
knowledge/concepts/world-model.md
knowledge/concepts/moving-mnist.md
knowledge/concepts/multi-task-learning.md
knowledge/concepts/spatial-encoder.md
knowledge/concepts/clustered-reservoir.md
knowledge/concepts/multi-timescale-dynamics.md
knowledge/concepts/input-routing.md
knowledge/concepts/linear-probe.md
knowledge/concepts/fine-tuning.md
knowledge/concepts/baselines.md
knowledge/concepts/ridge-regression.md
knowledge/concepts/neuroevolution.md

knowledge/connections/reservoir-paradigms.md
knowledge/connections/clustar-components.md
knowledge/connections/multi-task-self-supervision.md
knowledge/connections/evaluation-protocols.md
knowledge/connections/reservoir-world-models.md

knowledge/index.md
knowledge/log.md
```

---

**Compilation status**: Complete — all major concepts and connections extracted; ready for LaTeX export.
