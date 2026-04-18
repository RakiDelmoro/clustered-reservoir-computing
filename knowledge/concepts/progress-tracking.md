---
title: "Progress Tracking"
aliases: [progress bars, tqdm, visual feedback, ETA display]
tags: [user-experience, debugging, monitoring, long-running-jobs]
sources:
  - "daily/2026-04-16.md"  # Adding tqdm progress bars to feature extraction
created: 2026-04-16
updated: 2026-04-16
word_count: 260
---

# Progress Tracking

**Progress tracking** in CluSTAR refers to adding **visual progress bars** (via `tqdm`) to long-running operations, particularly the feature extraction phase in training. This provides real-time feedback on batch completion, estimated time remaining, and throughput.

## The Problem

Feature extraction runs reservoir forward pass on entire dataset:
- **Training**: 50,000 sequences / batch_size 64 = **782 batches**
- **Validation**: 10,000 sequences = **157 batches**
- Each batch: reservoir forward (1000 neurons) takes ~200-300ms on GPU
- **Total**: ~4-5 minutes of seemingly "frozen" terminal with no output

Without progress bars, users think the script **crashed or hung**.

## Solution: tqdm Wrapper

Wrap dataloader iteration:

```python
from tqdm import tqdm

# Before: silent loop
for batch in dataloader:
    features = extract(batch)

# After: progress bar
for batch in tqdm(dataloader, desc="Extracting features"):
    features = extract(batch)
```

**Output:**
```
Extracting features: 100%|██████████| 782/782 [04:06<00:00, 3.17it/s]
```

Shows:
- **Completion fraction** (100%)
- **Batch count** (782/782)
- **Elapsed time** (04:06)
- **ETA** (<00:00 when done)
- **Throughput** (3.17 batches/sec)

## Where Applied in CluSTAR

1. **`ActionClassifier.extract_features()`** — training and validation feature extraction
2. **`ActionClassifier.evaluate()`** — test set evaluation (157 batches)
3. **State collection** (if pre-training were used) — 200K unlabeled sequences

All inherit the progress bar automatically since they call `extract_features()`.

## Why tqdm Works Well

- **Zero-config**: auto-detects terminal width, disables in notebooks
- **Adaptive**: updates in-place without flooding terminal
- **Lightweight**: <1ms overhead per batch
- **Customizable**: `desc`, `total`, `unit`, `leave` parameters

## Best Practices for Research Code

1. **Always wrap long loops** (>10 sec) with tqdm
2. **Include descriptive `desc`**: `"Extracting train features"` vs `"Extracting val features"`
3. **Show throughput**: `tqdm` does this automatically (it/s)
4. **Don't leave=True** for nested bars (clutters terminal); default `leave=False` clears finished bars
5. **Position parameter** for parallel bars: `tqdm(dataloader, position=0)` vs `position=1`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Multiple bars overlapping | Use `tqdm(..., position=i)` for sequential bars |
| Not updating in Jupyter | Use `tqdm.notebook.tqdm` instead |
| No bar displayed (non-TTY) | tqdm auto-disables; force with `tqdm(..., disable=False)` |
| Too verbose | Set `miniters` or `mininterval` to reduce update frequency |

## Impact on CluSTAR

Adding tqdm **improved developer experience** significantly:
- Confirmed GPU was actually working (saw throughput 3.17 it/s vs CPU 2.5 it/s)
- Avoided premature termination during 4-minute extraction
- Provided concrete timing data for performance comparison

## Future Enhancements

- **Rich progress bars** with `tqdm-rich` or `alive-progress` for spinners
- **Per-epoch ETA** if adding iterative training (SGD instead of ridge)
- **Metric logging alongside bar**: `"Extracting (acc=0.92)"` via `set_postfix()`
- **Distributed training bars** with `tqdm.contrib.distributed`
