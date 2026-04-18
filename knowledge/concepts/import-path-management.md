---
title: "Import Path Management"
aliases: [package imports, PYTHONPATH, module resolution]
tags: [software-engineering, python, packaging, research-code]
sources:
  - "daily/2026-04-16.md"  # Fixing absolute imports with sys.path and clustar.* prefix
created: 2026-04-16
updated: 2026-04-16
word_count: 300
---

# Import Path Management

**Import path management** organizes Python's module search path (`sys.path`) so that intra-package imports resolve correctly when running scripts from different working directories. It's a common source of `ModuleNotFoundError` in research codebases with nested packages.

## The Problem Structure

CluSTAR package layout:
```
clustar/
├── data/
│   ├── __init__.py
│   ├── generator.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   └── reservoir.py
├── finetune/
│   ├── __init__.py
│   └── trainer.py
└── scripts/
    ├── run_training.py   # ← imports from data, models, finetune
    └── (no __init__.py)
```

**Script import statement:**
```python
from data.dataset import get_dataloaders
from models.encoder import SpatialEncoder
```

**What Python sees:**
- When running `python clustar/scripts/run_training.py` from inside `clustar/`:
  - `sys.path[0]` = `/workspaces/.../clustar` (script's directory)
  - Python looks for `data/` package **inside** `clustar/` → **fails** because `data` is a sibling of `scripts`, not a subdirectory

---

## Solution 1: Relative Imports (Standard)

Change scripts to use explicit relative imports:

```python
# Inside clustar/scripts/run_training.py
from ..data.dataset import get_dataloaders
from ..models.encoder import SpatialEncoder
from ..finetune.trainer import ActionClassifier
```

**Run command:**
```bash
cd /workspace  # parent of clustar/
python -m clustar.scripts.run_training --config clustar/configs/reservoir.yaml
```

**Why it works:** `-m clustar.scripts.run_training` treats `clustar` as a package; `..` means "go up to clustar package, then down into data/". Standard Python package behavior.

**Drawback:** Must always run from **parent directory** of `clustar/`. If you `cd clustar/` and run `python scripts/run_training.py`, relative imports fail with `ImportError: attempted relative import beyond top-level package`.

---

## Solution 2: Fully Qualified Absolute Imports + sys.path Hack (Used in CluSTAR)

Keep absolute imports but **prefix with package name** and add project root to `sys.path`:

```python
# clustar/scripts/run_training.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root

from clustar.data.dataset import get_dataloaders
from clustar.models.encoder import SpatialEncoder
from clustar.finetune.trainer import ActionClassifier
```

**Run from anywhere**:
```bash
cd clustar
python scripts/run_training.py --config configs/reservoir.yaml    # works
cd /workspace
python -m clustar.scripts.run_training --config clustar/reservoir.yaml  # also works
```

**How it works:**
- `sys.path.insert(0, project_root)` adds `/workspace` to module search path
- `clustar.data` becomes importable as top-level package
- No need to remember `-m` flag or special working directory

**Trade-off:** Slightly "hacky" but **user-friendly** for interactive development.

---

## Solution 3: Install Package in Editable Mode (Production)

Create `pyproject.toml` or `setup.py` at project root:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "clustar"
version = "0.1.0"
dependencies = ["torch", "numpy", ...]
```

Then:
```bash
pip install -e .
```

Now any script can do:
```python
from clustar.data.dataset import get_dataloaders
```
and Python finds `clustar` in site-packages (symlinked to your source tree). Works from any directory.

**Pros:** Clean, standard, works with IDEs, supports `pip install`
**Cons:** Requires packaging setup; slower iteration if you forget to reinstall after moves

---

## Solution 4: Entry Point Console Scripts

Define console script in `pyproject.toml`:

```toml
[project.scripts]
clustar-train = "clustar.scripts.run_training:main"
```

After `pip install -e .`, user runs:
```bash
clustar-train --config configs/reservoir.yaml
```

The `main()` function handles its own path setup or uses package-relative imports. Clean UX.

---

## Best Practice for Research Codebases

For **interactive research** (jupyter, rapid iteration), use **Solution 2** (`sys.path` hack + fully qualified imports):
- Minimal setup
- Works in notebooks and scripts
- Clear where imports come from (explicit `clustar.` prefix)
- No packaging overhead

For **public release / reproducibility**, add **Solution 3** (editable install) + optionally Solution 4 (console script). Provide both:
- `pip install -e .` for package mode
- `python scripts/run_training.py` for quick start (with sys.path fallback)

---

## CluSTAR's Approach

Combines **Solution 2** (sys.path + `clustar.*` imports):

```python
# scripts/run_training.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clustar.data.dataset import get_dataloaders
...
```

**Also added** `__init__.py` files to all package dirs (`data/`, `models/`, `finetune/`) to make them proper Python packages.

This ensures:
- ✓ Works when run as `python clustar/scripts/run_training.py` (from project root or any dir)
- ✓ Works when run as `python -m clustar.scripts.run_training`
- ✓ Works in Jupyter with `%run clustar/scripts/run_training.py`

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'data'` | Running script from inside package dir with absolute imports | Use `-m` flag or `sys.path` hack |
| `ImportError: attempted relative import beyond top-level package` | Using `..` in script run as `python script.py` | Use absolute imports + sys.path |
| Import resolves to wrong package | `sys.path` has wrong order (earlier entry shadows) | Print `sys.path` to debug ordering |
| Works locally but not in CI | CI runs from different working dir | Use `-m` or install package in CI |

---

## Debugging Import Errors

1. **Print `sys.path`** to see search order:
   ```python
   import sys
   print("\n".join(sys.path))
   ```
2. **Check `__init__.py`** exists in all parent packages (makes dir a package)
3. **Use absolute import with package prefix** — easier to trace than relative
4. **In CI/CD**: use `python -m` or `pip install -e .` — never rely on cwd

---

## Conclusion

Import path issues are **universal** in Python projects. CluSTAR's solution (sys.path hack + `clustar.` prefix) is pragmatic for research code. For production, migrate to editable install with console scripts.
