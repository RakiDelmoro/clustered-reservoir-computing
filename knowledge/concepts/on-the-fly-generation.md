---
title: "On-the-Fly Generation"
aliases: [lazy generation, runtime synthesis, memory-efficient datasets]
tags: [data-engineering, memory-optimization, moving-mnist, dataset-design]
sources:
  - "daily/2026-04-16.md"  # Switching from pre-caching to on-the-fly sequence generation
created: 2026-04-16
updated: 2026-04-16
word_count: 300
---

# On-the-Fly Generation

**On-the-fly generation** is a dataset design pattern where sequences are synthesized **at runtime** when accessed, rather than pre-computed and stored in memory. This drastically reduces RAM requirements for large synthetic datasets.

## The Problem: Pre-Caching All Sequences

Initial CluSTAR implementation:

```python
class MovingMNISTDataset:
    def __init__(self, num_samples=50000, seq_length=30):
        self.sequences = []  # Pre-generate all
        for i in range(num_samples):
            seq = generator.generate_sequence(...)
            self.sequences.append(seq)
```

**Memory cost:**
- Each sequence: `30 × 64 × 64 × float32` = ~30 KB
- 50K sequences: 50,000 × 30 KB = **~1.5 GB** (frames only)
- With metadata, trajectories, multiple digits → **~5-10 GB**
- Pre-training with 200K unlabeled: **200K × 30 KB = 6 GB frames** → total ~20-30 GB

For research laptops or embedded targets, this is **prohibitively large**.

---

## Solution: Generate on Access

```python
class MovingMNISTDataset:
    def __init__(self, generator, num_samples, seq_length, seed):
        self.generator = generator  # Shared, lightweight
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.base_seed = seed  # For deterministic per-index generation

    def __getitem__(self, idx):
        # Deterministic RNG per index (same sequence every time)
        rng = np.random.RandomState(self.base_seed + idx)
        seq = self.generator.generate_sequence(
            action=rng.choice(self.actions),
            seq_length=self.seq_length,
            rng=rng
        )
        return seq
```

**Memory now:** Only digit templates (10 × 28×28 = ~7 KB) + generator state. Negligible.

**Trade-off:** CPU cost at access time (~2-5ms per sequence on modern CPU; ~100-200μs on GPU if needed). For 50K samples during training, total generation cost ~2-4 minutes — **comparable to forward pass time anyway**, so no net slowdown.

---

## Ensuring Determinism

On-the-fly generation **must be deterministic** — same index → same sequence across epochs.

```python
def __getitem__(self, idx):
    rng = np.random.RandomState(self.base_seed + idx)
    # All randomness derived from this RNG → reproducible
```

Without this, training would see different sequences each epoch → no convergence.

---

## Parallel Data Loading

When using `DataLoader(num_workers > 0)`, each worker process gets its own copy of the generator. Ensure generator state is **picklable** or re-initialized per worker:

```python
def worker_init_fn(worker_id):
    # Each worker sets its own base seed offset
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
```

For CluSTAR, `MovingMNISTGenerator` is picklable (holds digit templates as numpy arrays), so multi-worker loading works.

---

## When to Use On-the-Fly

| Scenario | On-the-fly recommended? | Reason |
|----------|------------------------|--------|
| **Small discrete datasets** (CIFAR-10) | No — load once from disk | Simpler |
| **Large synthetic datasets** (Moving MNIST 200K) | Yes — RAM prohibitive otherwise | Memory savings |
| **Highly augmentable data** (images with random transforms) | Often on-the-fly augment *plus* generation | Both |
| **Pre-trained feature datasets** (pre-computed ResNet features) | No — compute once, reuse | Expensive feature extraction |
| **Real-time simulation** (robotics, games) | Yes — environment steps on-demand | Environment IS the generator |

---

## Trade-offs

**Pros:**
- **RAM efficient**: O(1) memory vs O(N) for pre-cache
- **Infinite data**: can generate arbitrary number of unique sequences
- **Dynamic difficulty**: curriculum learning by varying generation params at runtime
- **Versioning**: generator code changes automatically affect all samples

**Cons:**
- **CPU overhead** per sample (~few ms)
- **Reproducibility complexity**: must manage seeds across workers
- **No pre-fetch benefit**: can't pre-load to disk for faster sequential reads
- **Harder to inspect**: can't browse saved samples without re-generating

---

## Implementation in CluSTAR

**Files modified:**
- `data/generator.py` — unchanged; generator already supports per-call RNG
- `data/dataset.py` — converted from pre-caching to `__getitem__` generation
- `test_sanity.py` — uses on-the-fly; no caching

**Memory impact:**
```
Before: 50K sequences → ~24 GB RAM ❌
After:  digit templates (10×28×28) → ~70 KB ✅
```

**Speed impact:** Negligible because generation time (~3ms/seq) ≈ reservoir forward pass time (~5ms/seq on GPU). They're **pipeline parallel**: next batch generates while current batch forwards.

---

## Generalization to Other Domains

On-the-fly pattern works for:
- **Synthetic videos** (moving shapes, bouncing balls)
- **Procedural environments** (MinAtar, Procgen)
- **Audio synthesis** (waveform generation)
- **Text augmentation** (synonym replacement, back-translation)
- **3D rendering** (Blender, mujoco collisions)

Key requirement: **generator must be fast and deterministic**.
