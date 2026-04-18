---
title: "Device Management"
aliases: [GPU/CPU placement, tensor device, device mismatch]
tags: [debugging, performance, gpu, pytorch, device-handling]
sources:
  - "daily/2026-04-16.md"  # GPU device mismatch bug fixes (alpha buffer, label-device alignment)
created: 2026-04-16
updated: 2026-04-16
word_count: 400
---

# Device Management

**Device management** in PyTorch ensures all tensors and modules reside on the same device (CPU or GPU) during computation. Mismatches cause `RuntimeError: Expected all tensors to be on the same device`.

## Common Device Mismatch Patterns

### Pattern 1: Unregistered Buffers Don't Move with `.to(device)`

**Problem:**
```python
class Reservoir(nn.Module):
    def __init__(self, size):
        self.alpha = torch.zeros(size)  # Plain tensor, NOT registered
```
When calling `reservoir.to('cuda')`, `alpha` **stays on CPU** because it's not a buffer or parameter.

**Symptoms:**
```
RuntimeError: Expected all tensors to be on the same device, 
but found at least two devices, cuda:0 and cpu!
```
Error occurs in operations involving `alpha` (e.g., `x * alpha`).

**Fix:**
```python
self.register_buffer("alpha", torch.zeros(size))
# OR (for version <1.10):
self.alpha = torch.zeros(size)  # Then manually: self.alpha = self.alpha.to(device)
```

**CluSTAR application:** `models/reservoir.py:79` — fixed `alpha` leaking rates as buffer.

---

### Pattern 2: Labels and Targets on Different Devices

**Problem:**
```python
X_train = ...  # on GPU after extract_features with device='cuda'
y_train = labels  # from dataloader, stays on CPU
y_onehot = torch.zeros(...).to(device)  # on GPU
y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)  # CPU indices → GPU tensor ❌
```

**Fix:**
```python
# Move labels to target device before in-place ops
y_train = y_train.to(device)
# OR: do scatter on CPU, then move result
y_onehot = torch.zeros(..., device='cpu')
y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
y_onehot = y_onehot.to(device)
```

**CluSTAR fix:** `finetune/trainer.py:161` — moved `y_train` to CPU for scatter, then moved `W, b` back to GPU after ridge solve.

---

### Pattern 3: Mixed Device in DataLoader

By default, DataLoader returns tensors on **CPU**. If you want GPU tensors immediately:

```python
# Option A: Move after collection
X, y = extract_features(dataloader)  # CPU
X, y = X.to(device), y.to(device)

# Option B: Pin memory + non-blocking (faster host→device transfer)
dataloader = DataLoader(..., pin_memory=True)
X = X.to(device, non_blocking=True)
```

**CluSTAR choice:** Extract on CPU (simpler), then move features to GPU only for visualization/test (small batches).

---

## Device Strategy in CluSTAR

| Component | Device | Rationale |
|-----------|--------|-----------|
| **Spatial Encoder** | CPU or GPU (matches reservoir) | Small matrix multiply (128×4096); negligible |
| **Reservoir forward** | **GPU** (if available) | 1000×1000 matrix multiply per timestep; GPU 10-30× faster |
| **State collection** | GPU for reservoir, CPU accumulation | States collected on GPU then moved to CPU to avoid OOM |
| **Ridge regression** | **CPU** (numpy-like) | Closed-form solve involves large matrix inversion (T×T or N×N); CPU BLAS optimal; memory not fit on GPU for 50K×1000 |
| **Readout inference** | GPU (if reservoir on GPU) | Final `W_out @ x` is tiny, GPU overhead dominates — can stay CPU |

**Actual pipeline:**
1. Run reservoir forward on GPU (batched)
2. Collect states → move to CPU (concatenate all 50K)
3. Solve ridge on CPU (scipy/numpy ops via torch)
4. Move final weights W, b to GPU for evaluation (optional)

---

## Debugging Checklist

When you see `RuntimeError: Expected all tensors to be on the same device`:

1. **Print device of every tensor** involved in the operation:
   ```python
   print(f"X.device={X.device}, y.device={y.device}, W.device={W.device}")
   ```
2. **Check `.to(device)` calls** — did you forget one?
3. **Check for buffers** (like `alpha`) — are they registered with `register_buffer`?
4. **Check model initialization** — are you constructing new tensors inside `forward()` without respecting device?
   ```python
   # Bad:
   def forward(self, x):
       bias = torch.zeros(self.size)  # Always CPU!
   # Good:
   bias = torch.zeros(self.size, device=x.device)
   ```
5. **Check DataLoader collate_fn** — does it return CPU tensors? (usually yes)

## Utility Functions

Centralize device logic:

```python
def to_device(tensor, device):
    """Move tensor or dict/list of tensors to device."""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        return {k: to_device(v, device) for k, v in tensor.items()}
    elif isinstance(tensor, list):
        return [to_device(t, device) for t in tensor]
    return tensor

def assert_same_device(*tensors):
    devices = [t.device for t in tensors if hasattr(t, 'device')]
    assert len(set(devices)) == 1, f"Devices differ: {devices}"
```

## Performance Implications

- **GPU↔CPU transfer** is expensive (~1-10ms per transfer). Minimize frequency.
- **Batch extraction on GPU** → collect → single transfer to CPU is optimal.
- **Ridge solve on CPU** avoids GPU RAM limits (50K×1000 matrix = 200MB float32; GPU might OOM if also storing reservoir states).
- **Mixed-precision** (FP16) doesn't help ridge solve (needs FP32 for stability).

## Lessons from CluSTAR

The device mismatches encountered during development were:
1. **`alpha` buffer** — learned to always use `register_buffer` for fixed parameters
2. **Label scatter** — learned to align index tensor with destination tensor device
3. **Import order** — moving imports after device assignment sometimes triggers earlier initialization on wrong device

These are **common pitfalls** in any PyTorch research codebase; systematic checks prevent them.
