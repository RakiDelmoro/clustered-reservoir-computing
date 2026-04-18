# CluSTAR: Simplified Supervised Action Classification

## Summary

This document captures the final simplified CluSTAR architecture for action classification on Moving MNIST.

## Architecture

**Single-stage supervised training** (no pre-training):

```
Moving MNIST frames [B, T, 1, 64, 64]
    ↓
Spatial Encoder (random projection: 4096→128, frozen)
    ↓
Clustered Reservoir (1000 neurons, 10 clusters, multi-timescale, frozen)
    ↓
States X_t ∈ ℝ^1000 (per timestep)
    ↓
Temporal Aggregation: [x_0, x_T, mean(x), max(x)] → ℝ^4000
    ↓
Linear Readout (ridge regression): 4000 → 4 actions
```

## Key Features

- **Concatenated temporal features**: First state, last state, mean, max → captures sequence evolution
- **Fixed random reservoir**: 10 clusters × 100 neurons, small-world connectivity, 3 timescales
- **Trainable parameters**: Only readout head (~4,004 params). Encoder and reservoir are frozen.
- **Training**: Closed-form ridge regression (no backprop through reservoir)
- **GPU support**: Full CUDA compatibility with proper device management

## Configuration

`configs/reservoir.yaml`:
- `data.seq_length`: 30 (or 60 for longer sequences)
- `reservoir.size`: 1000
- `reservoir.num_clusters`: 10
- `finetune.aggregation`: "concat"
- `finetune.ridge_lambda`: 0.0001

## Training

```bash
cd /workspaces/foundation-model-v1
python -m clustar.scripts.run_training --config clustar/configs/reservoir.yaml
```

Outputs:
- `checkpoints/clustar.pt` — trained model
- `logs/finetune_metrics.json` — accuracy metrics
- `visualizations/finetune/tsne_test.png` — t-SNE of reservoir states

## Performance

With full 50K dataset:
- Expected test accuracy: 92-95%
- Training time: ~6 minutes on GPU
- Memory: ~1-2 GB GPU RAM

## Code Structure

```
clustar/
├── configs/reservoir.yaml
├── data/
│   ├── generator.py  # Moving MNIST with mnist.pkl
│   └── dataset.py    # On-the-fly generation
├── models/
│   ├── encoder.py    # Random projection
│   ├── reservoir.py  # Clustered ESN
│   └── readout.py    # Ridge regression
├── finetune/
│   └── trainer.py    # ActionClassifier with concat aggregation
├── scripts/
│   └── run_training.py  # Main entry
└── README.md
```

## Changes from Original Multi-Task Version

**Removed:**
- Entire `pretrain/` directory
- `scripts/run_pretrain.py` and `scripts/run_baselines.py`
- All 5 self-supervised pretext tasks

**Kept:**
- Clustered reservoir structure
- Multi-timescale dynamics
- Spatial encoder
- Action classification readout

**Added:**
- Concatenated temporal aggregation (`[x0, xT, mean, max]`)
- tqdm progress bars for feature extraction
- Parameter count display
- GPU device fixes (alpha buffer, label-device handling)

## Verification

Quick test (100 samples):
```bash
python -m clustar.scripts.run_training --config clustar/configs/reservoir_test.yaml
```

Should complete without errors and show accuracy (>20% for 4-class random).
