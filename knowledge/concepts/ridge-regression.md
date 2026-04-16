---
title: "Ridge Regression"
aliases: [Tikhonov regularization, closed-form linear regression]
tags: [linear-algebra, training-algorithm, readout-layer]
sources:
  - "daily/2026-04-16.md"  # Ridge regression readout training
created: 2026-04-16
updated: 2026-04-16
word_count: 220
---

# Ridge Regression

**Ridge regression** (Tikhonov regularization) is the training algorithm for ESN and CluSTAR readout layers. It solves a regularized linear system in closed form, avoiding iterative gradient descent.

## Problem Formulation

Given:
- Reservoir states matrix X ∈ ℝ^(N×T) (N neurons, T timesteps)
- Target outputs Y ∈ ℝ^(C×T) (C output dimensions)

Find linear weights W ∈ ℝ^(C×N) minimizing:
```
L(W) = ||Y - W·X||²_F + λ·||W||²_F
```
where λ > 0 is L2 regularization parameter.

## Closed-Form Solution

Normal equations with regularization:
```
W = Y·X^T · (X·X^T + λ·I_N)^{-1}
```

Equivalent to solving `(X·X^T + λI)w_c = x_c` for each output dimension c separately.

**Why invert N×N not T×T?** N (reservoir size, ~1000) << T (timesteps, ~millions) → cheaper.

## Ridge vs. Ordinary Least Squares (OLS)

- **OLS**: λ=0, solution may not exist if X^T X singular or ill-conditioned
- **Ridge**: λ>0 ensures unique, stable solution; reduces overfitting

λ chosen via cross-validation on validation set (typical: 1e-6 to 1e-1).

## Multi-Task Ridge

For M tasks with targets Y₁, ..., Y_M:
```
W = [W₁; W₂; ...; W_M] = [Y₁·X^T; Y₂·X^T; ...] · (X·X^T + λI)^{-1}
```
Same X for all tasks; concatenated outputs → single solve.

## Advantages for Reservoir Computing

1. **No backpropagation** — compute X once, solve linear system
2. **Fast** — O(N³) matrix inversion, but N~1000 is trivial on modern CPU/GPU
3. **Deterministic** — same X, λ → same solution every run
4. **Online variant** — recursive least squares (RLS) for streaming data
5. **Sparse solutions** — can use L1 (lasso) but ridge sufficient for RC

## Numerical Stability

Direct inversion can be unstable if X·X^T ill-conditioned. Fixes:
- Use `np.linalg.solve` (factorization) instead of explicit inverse
- Add small jitter (1e-8) to diagonal
- Cholesky decomposition for positive-definite matrix

## In CluSTAR Code

Implemented in `clustar/models/readout.py`:
```python
class RidgeRegression:
    def solve(self, X, Y, lambda_=1e-4):
        # X: [N, T], Y: [C, T]
        A = X @ X.T + lambda_ * torch.eye(N)
        b = X @ Y.T
        W = torch.linalg.solve(A, b).T  # [C, N]
        return W
```

Used for:
- Pretext task heads training (batch-collected X)
- Downstream action classification
- Baselines (vanilla ESN, LSTM final layer)
