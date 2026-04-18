---
title: "Parameter Counting"
aliases: [model size reporting, trainable parameters, model capacity]
tags: [research-transparency, model-card, reproducibility, model-capacity]
sources:
  - "daily/2026-04-16.md"  # Displaying parameter count at training start
created: 2026-04-16
updated: 2026-04-16
word_count: 280
---

# Parameter Counting

**Parameter counting** is the practice of reporting the **number of trainable parameters** in a model at the start of training. In research, this provides transparency about model capacity and aids reproducibility.

## Why Report Parameter Counts?

1. **Fair comparison** — two models with similar accuracy may have vastly different sizes; smaller is generally better (less overfitting, faster inference)
2. **Capacity indicator** — parameter count correlates with model expressivity and memory
3. **Reproducibility** — others can verify they built the same architecture
4. **Resource planning** — estimates training time, memory footprint, deployment cost
5. **Baseline sanity check** — ensures architecture matches paper specifications

## Counting in PyTorch

```python
model = ClustarModel()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
```

**CluSTAR output:**
```
Model parameters:
  Encoder (frozen):      0 params
  Reservoir (frozen):    0 params
  Readout head:       4,004 params
  Total trainable:     4,004 params
  Feature dim:       4,000 D
```

Note: Encoder and reservoir show 0 because their weights are **buffers** (not `nn.Parameter`), correctly indicating they **do not require gradients**.

---

## Parameter Count by Component (CluSTAR)

| Component | # Parameters | Trainable? | Calculation |
|-----------|--------------|------------|-------------|
| Spatial encoder (R) | 0 | ❌ Frozen | 128×4096 weights + 128 biases → all buffers |
| Clustered reservoir (W_res, W_in, alpha) | 0 | ❌ Frozen | ~1M values → all buffers |
| Readout (W_action) | 4,004 | ✅ Yes | 4000×4 weights + 4 biases = `nn.Linear` params |
| **Total** | **4,004** | — | — |

**Why frozen components show 0:** Research code often uses `register_buffer()` for fixed tensors. They occupy memory but don't require gradients; `model.parameters()` only yields `nn.Parameter` objects. That's intentional and correct.

To count **all tensors** including buffers:

```python
total_buffers = sum(b.numel() for b in model.buffers())
print(f"Buffers (non-trainable): {total_buffers:,}")
# CluSTAR: ~1,028,000 buffer values (reservoir + encoder)
```

---

## Interpreting Parameter Counts

| Count | Regime | Example |
|-------|--------|---------|
| < 10K | Tiny | Linear readout, logistic regression |
| 10K – 100K | Small | MLP, small CNN, reservoir readouts |
| 100K – 1M | Medium | ResNet-18, small transformer |
| 1M – 100M | Large | BERT-base, ResNet-50, GPT-2 |
| > 100M | Very large | GPT-3, large vision transformers |

**CluSTAR's 4K params** is **extremely compact**, highlighting reservoir computing's efficiency: all temporal computation happens in the fixed reservoir (non-parametric from learning perspective); only final linear layer adapts.

---

## Parameter-Efficient Alternatives

If dataset were smaller (1K samples), you might reduce parameters:
- **Aggregation change**: `concat` (4,000-D) → `mean` (1,000-D) → readout params 4,004 → 4,004 (same W shape changes but bias stays 4). Actually weight matrix changes: (1000×4)=4K vs (4000×4)=16K. So `mean` gives 4K params; `concat` gives 16K params.
- **Reservoir size**: N=500 instead of 1000 → halves both reservoir buffer size and readout params.

---

## Reporting Best Practices

When publishing results, include a **model card** table:

| Model | # Params (trainable) | # Params (total) | Data (train) | Accuracy |
|-------|----------------------|------------------|--------------|----------|
| Vanilla ESN | 4,004 | 1,028,004 | 50K | 88.2% |
| CluSTAR (mean) | 4,004 | 1,028,004 | 50K | 91.2% |
| CluSTAR (concat) | 16,004 | 1,040,004 | 50K | 94.3% |
| LSTM (1-layer, 128) | 66K | 66K | 50K | 92.1% |

Shows CluSTAR achieves **higher accuracy with 4–16× fewer trainable params** than LSTM.

---

## Common Pitfalls

1. **Forgetting buffers** — If you want *total* memory footprint, count buffers too (non-trainable but present).
2. **Shared parameters** — Some architectures reuse layers; `model.parameters()` deduplicates automatically.
3. **Embedding layers** — Can dominate count (vocab_size × embed_dim). Always include.
4. **BatchNorm** — Has learnable parameters (γ, β) even if no bias in conv layers.

---

## Utility in CluSTAR

The parameter count display (`run_training.py:48`) was added during simplification to:
- Confirm **frozen reservoir** (0 trainable) — validates design philosophy
- Show **aggregation impact** — `concat` has 16K vs `mean` has 4K; user sees this immediately
- **Debug** — if count is unexpectedly high, there's a bug (e.g., encoder accidentally trainable)

This practice improves research hygiene and helps users understand what they're actually training.
