---
title: "Script Organization"
aliases: [pipeline design, training scripts, codebase structure]
tags: [software-engineering, research-code, workflow, maintainability]
sources:
  - "daily/2026-04-16.md"  # Consolidating multi-script pipeline to single run_training.py
created: 2026-04-16
updated: 2026-04-16
word_count: 300
---

# Script Organization

**Script organization** refers to how training/inference workflows are structured across multiple command-line entry points. Research codebases often evolve from **single-script simplicity** to **multi-script pipelines** as features grow, then sometimes simplify back.

## The Evolution

CluSTAR script structure evolved:

```
Phase 1: Multi-script (complex)
├── scripts/run_pretrain.py      # 200K unlabeled pre-training
├── scripts/run_finetune.py      # 50K labeled fine-tuning
├── scripts/run_baselines.py     # Vanilla ESN, LSTM comparisons
└── configs/pretrain.yaml, finetune.yaml, reservoir.yaml

Phase 2: Simplified (single-stage)
├── scripts/run_training.py      # Single supervised training (replaces all above)
├── configs/reservoir.yaml       # One config to rule them all
└── (pretrain/, run_baselines.py deleted)
```

---

## Multi-Script Pipeline (Original)

**Advantages:**
- Separation of concerns: pre-train vs. finetune clearly separated
- Can run pre-training once, then multiple fine-tune experiments with different heads
- Parallelizable: pre-training on GPU A, fine-tuning on GPU B
- Config modularity: separate hyperparameters per phase

**Disadvantages:**
- **Cognitive overhead**: users must understand 3+ scripts and their relationships
- **More files to maintain**: each script needs its own CLI args, logging, error handling
- **Config sprawl**: multiple YAML files (pretrain.yaml, finetune.yaml, reservoir.yaml)
- **Integration bugs**: pretrain output format must match finetune input format exactly
- **Harder to reproduce**: full experiment requires running multiple scripts in sequence

---

## Single-Script Pipeline (Simplified CluSTAR)

**Advantages:**
- **One command**: `python -m clustar.scripts.run_training --config configs/reservoir.yaml`
- **Single source of truth**: all hyperparameters in one YAML
- **Easier to modify**: add option → one argparse block
- **Simpler CI/CD**: one entry point to test
- **Lower barrier to entry**: new users aren't confused by multi-phase complexity

**Disadvantages:**
- **Less modular**: hard to reuse pre-trained representations across different downstream tasks without re-running full pipeline
- **Less flexible**: can't swap pre-training dataset separately from fine-tuning dataset without config changes
- **Single point of failure**: script complexity concentrated in one file

---

## Hybrid Approach (Recommended for Larger Projects)

Keep **one master script** but modularize internally:

```python
# run.py
def pretrain(args):
    """Self-supervised pre-training phase."""
    pass

def finetune(args):
    """Downstream fine-tuning phase."""
    pass

def train_supervised(args):
    """Single-stage supervised training."""
    pass

def evaluate(args):
    """Evaluation only."""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('pretrain')
    subparsers.add_parser('finetune')
    subparsers.add_parser('train')
    subparsers.add_parser('eval')
    args = parser.parse_args()
    if args.command == 'pretrain': pretrain(args)
    ...
```

**User commands:**
```bash
python run.py pretrain --config pretrain.yaml
python run.py finetune --config finetune.yaml
python run.py train --config reservoir.yaml
python run.py eval --checkpoint clustar.pt
```

This is what **CluSTAR originally had** (separate scripts) but could be consolidated into one file with subcommands.

---

## Config File Organization

Corresponding to script organization:

| Script Org | Config Style | Example |
|------------|--------------|---------|
| Multi-script | Multiple YAMLs | `pretrain.yaml`, `finetune.yaml`, `reservoir.yaml` |
| Single-script | Unified YAML | `reservoir.yaml` with `pretrain: {}` and `finetune: {}` sections |
| Hybrid | Config inheritance | `base.yaml` + `pretrain.yaml` (extends base) |

CluSTAR simplification chose **single YAML** with only the `finetune` section (no `pretrain`).

---

## Research Codebase Recommendations

**For < 3 training modes:** Single script is fine.
- `scripts/train.py` with `--mode pretrain|finetune|eval`
- `configs/default.yaml` with optional sections

**For > 3 modes or heavy reuse:** Hybrid with subcommands.
- `scripts/run.py pretrain ...`
- `scripts/run.py finetune ...`
- `scripts/run.py export ...`

**For production pipelines:** Full orchestrator (e.g., `snakemake`, `luigi`, `prefect`), but overkill for research.

---

## CluSTAR's Final Choice

**Single script: `scripts/run_training.py`** (originally `run_finetune.py`)

**Reasons:**
- Single downstream task (action classification)
- No pre-training needed (simplified research question)
- Easy for others to reproduce: one file to read, one command to run
- Aligns with **minimal viable codebase** philosophy

**If pre-training were re-added,** the script would split back into:
- `scripts/run_pretrain.py` — collects states, trains 5 readouts
- `scripts/run_finetune.py` — loads pre-trained readout, trains action head

But for now, **simplicity wins**.

---

## Improving Discoverability

Add a `Makefile` or `justfile` for common commands:

```makefile
# Makefile
train:
    python -m clustar.scripts.run_training --config configs/reservoir.yaml

test:
    python test_sanity.py

clean:
    rm -rf checkpoints/ logs/ visualizations/

viz:
    python visualize_moving_mnist.py
```

Now users can run `make train` instead of remembering the full command.

**Alternative:** `justfile` (simpler syntax, no Make idiosyncrasies).

This improves UX without adding script complexity.
