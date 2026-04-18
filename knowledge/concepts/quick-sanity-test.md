---
title: "Quick Sanity Test"
aliases: [smoke test, minimal validation, development iteration]
tags: [debugging, development-workflow, testing, iteration-speed]
sources:
  - "daily/2026-04-16.md"  # Running with 100 samples to verify end-to-end before full training
created: 2026-04-16
updated: 2026-04-16
word_count: 240
---

# Quick Sanity Test

A **quick sanity test** (or smoke test) is a minimal, fast-running validation of the full training pipeline using a **tiny subset** of data. It catches integration errors before committing to hours of full-scale training.

## Why Run Sanity Tests?

Full CluSTAR training on 50K sequences takes **5-6 minutes** on GPU. If there's a bug, you only discover it after 5 minutes. Sanity test runs in **10-30 seconds** on 100 samples, providing rapid iteration.

**What it validates:**
- ✅ Data generator produces valid sequences (no NaN, correct shape)
- ✅ Encoder outputs correct dimensions
- ✅ Reservoir forward pass completes without shape mismatches
- ✅ Feature extraction produces expected tensor shapes
- ✅ Ridge regression solver receives correct tensor shapes
- ✅ Evaluation metrics compute without errors
- ✅ GPU device transfers work (if using CUDA)
- ✅ Model checkpoint saving/loading

**What it does NOT validate:**
- ❌ Convergence (100 samples too few for meaningful accuracy)
- ❌ Generalization (test accuracy will be random ~25%)
- ❌ Hyperparameter optimality (λ, reservoir size not tuned)

---

## Typical Sanity Test Configuration

**Original config** (`reservoir.yaml`) modified:

```yaml
data:
  train_samples: 100    # was 50000
  val_samples: 20       # was 10000
  test_samples: 20      # was 10000

training:
  batch_size: 16       # smaller for tiny dataset
  device: "cuda"       # or "cpu"
```

**Command:**
```bash
python -m clustar.scripts.run_training --config clustar/configs/reservoir_small.yaml
```

**Expected output:**
- Train accuracy: ~100% (ridge can fit tiny training set perfectly)
- Val accuracy: ~15-35% (high variance due to tiny val set)
- Test accuracy: ~20-30% (near random for 4 classes)
- Model saves without errors

---

## Sanity Test Checklist

After running, verify:

- [ ] **No exceptions** raised during data loading
- [ ] **Progress bars** appear and complete (not frozen)
- [ ] **Shapes match**: features `[100, 4000]`, labels `[100]`
- [ ] **Checkpoint file** created in `checkpoints/` (non-zero size)
- [ ] **Logs** written to `logs/`
- [ ] **Visualizations** generated (if enabled)
- [ ] GPU memory **doesn't OOM** (if using CUDA)

If any fail → fix before full training.

---

## Iteration Speed Comparison

| Iteration | Full Training (50K) | Sanity Test (100) |
|-----------|--------------------|-------------------|
| Bug discovery | 5 min wait | 15 sec wait |
| Fix → re-test | 5 min | 15 sec |
| **Cycle time** | ~5 min | **~15 sec** |
| **Per-day iterations** | ~12 cycles | ~240 cycles |

**120× faster iteration** with sanity tests. Critical for active development.

---

## Advanced: Parameter-Efficient Sanity

Sometimes you want even faster (~5 sec) to debug shape errors:

```python
# test_sanity.py — no training, just forward pass
from clustar.models import SpatialEncoder, ClusteredReservoir
encoder = SpatialEncoder()
reservoir = ClusteredReservoir()
dummy = torch.randn(2, 30, 1, 64, 64)  # 2 sequences
states = reservoir.forward_sequence(encoder(dummy))
assert states.shape == (2, 30, 1000)
print("✓ Shapes correct")
```

This validates architecture wiring without data loading or training.

---

## When to Skip Sanity Tests

- **One-off experiments** where you're confident the code works
- **Final runs** for paper results (always use full data)
- **Hyperparameter search** where config changes are minor and already validated

**Rule of thumb:** Always run sanity test after modifying:
- Data generator
- Reservoir architecture
- Aggregation method
- Import paths / package structure
- Device handling logic

---

## Example: CluSTAR Sanity Test Session

From the log, we ran:

```bash
python3 test_sanity.py
```

Output:
```
[INFO] Testing reservoir forward...
[INFO] Extracting features...
[INFO] Training ridge regression...
[INFO] Test accuracy: 24.69%
[INFO] Sanity test passed!
```

24.69% ≈ random (25%) — expected with 10 training samples, no pre-training. The fact that it **doesn't crash** is the success criterion.

---

## Sanity Test vs. Unit Test

| | Sanity Test | Unit Test |
|---|-------------|-----------|
| Scope | Full pipeline integration | Single function/class |
| Data | Tiny real dataset | Mock data or fixtures |
| Speed | ~10-30 sec | ~<1 sec |
| Frequency | Every code change before full run | Every commit |
| Automation | Manual run | Part of CI/CD |

**Best practice:** Have both. Sanity test for integration; unit tests for correctness of components.

---

## Best Practices

1. **Keep sanity config separate** — `reservoir_small.yaml` committed to repo
2. **Make it fast** — use minimal `seq_length=10` if possible
3. **Make it deterministic** — fixed seed so failures reproducible
4. **Add assertions** — check shapes, NaN, device placement
5. **Document** — README should say "Run sanity test before full training"

The CluSTAR `test_sanity.py` script implements all these.
