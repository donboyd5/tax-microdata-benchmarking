**Read `repo_conventions_session_notes.md` first.**

# Draft GitHub Issue: Switch reweighting solver

Copy everything below the line into a GitHub issue.

---

## Title: Improve cross-machine reproducibility, quality, and speed of national weight optimization by reformulating as QP and switching to scipy L-BFGS-B

@martinholmer, any comments on the following?

Prepared with the help of Claude code.

## Summary

We can improve reproducibility, quality, and speed of national weight optimization by reformulating the problem as a quadratic program (QP) and switching the reweighting solver from PyTorch L-BFGS to scipy L-BFGS-B. We can do this in a non-disruptive manner by waiting until we update optimization targets to 2022 values.

## Problem

Reweighting is not reproducible across machines. Two different computers — even two different CPUs — can produce materially different weights (`np.allclose()` fails). The most dramatic case is GPU vs CPU, but the problem affects any two environments with different floating-point behavior (different CPU architectures, different BLAS libraries, different OS/compiler combinations).

The root cause is structural: PyTorch L-BFGS uses `torch.clamp()` to enforce bound constraints, which creates flat gradient regions where the optimizer cannot distinguish between solutions. On this flat loss surface, floating-point non-associativity in `.sum()` reduction operations causes different hardware to push records to opposite bound extremes (e.g., multiplier 0.1 on one machine, 10.0 on another for the same record).

## Key Insight: The Existing Objective Is Already a Convex QP

The reweighting objective function is:

```
minimize  sum_j [ (sum_i w0_i * m_i * A_ij) / (t_j + 1) - t_j / (t_j + 1) ]^2
        + penalty * L0 * sum_i (w0_i^2 / sum(w0^2)) * (m_i - 1)^2

subject to:  m_min <= m_i <= m_max
```

where `m_i` are weight multipliers, `w0_i` are original weights, `A_ij` is the output matrix, `t_j` are SOI targets, and `L0` is the initial unpenalized loss (for scaling the penalty term).

This is a **bound-constrained convex quadratic program (QP)**. The Hessian is positive definite, which guarantees a **unique global minimum**. We have been solving this problem all along — but with PyTorch L-BFGS, we were not exploiting its QP structure:

- **Clamping instead of projected gradients**: `torch.clamp()` zeros out gradients at bounds, creating flat regions where the optimizer stalls. A proper projected-gradient method (like L-BFGS-B) projects the gradient onto the feasible set, maintaining useful gradient information at bounds.
- **Autograd instead of analytical gradient**: The gradient of a QP is a simple linear function. PyTorch's autograd computes it correctly but introduces unnecessary overhead and floating-point variability compared to the closed-form expression.
- **No uniqueness guarantee exploited**: Because clamping creates equivalent-loss plateaus at bounds, the PyTorch solver can settle at different points on the plateau depending on hardware-specific floating-point behavior. A solver that properly handles bounds converges to the unique minimum regardless of platform.

## Proposed Two-Part Solution

### Part 1: Reformulate the problem as a QP and switcch to L-BFGS-B

Replace the PyTorch L-BFGS solver with **scipy's `L-BFGS-B`** (Fortran implementation), which properly exploits the QP structure:

1. **Projected-gradient bounds** — no clamping artifacts, no flat gradient regions
2. **Analytical gradient** — closed-form QP gradient, deterministic and efficient:
   ```
   grad_i = 2 * sum_j B_ij * (sum_k B_kj * m_k - c_j) + 2 * lam_i * (m_i - 1)
   ```
   where `B_ij = w0_i * A_ij / (t_j + 1)` and `c_j = t_j / (t_j + 1)`
3. **Same objective function** — no change to what we're optimizing, only how
4. **Unique minimum convergence** — any machine converging tightly enough must find the same answer

#### Experimental Results

Tested on branch `experiment-grad-norm` (preserved on fork):

| Solver | Loss | Time | Converged | Cross-machine reproducible |
|--------|------|------|-----------|---------------------------|
| PyTorch L-BFGS (GPU) | 0.1126 | ~90s | Stagnated | No |
| PyTorch L-BFGS (CPU) | 0.1126 | ~180s | Stagnated | No |
| **Scipy L-BFGS-B** | **0.1065** | **422s** | **Yes (grad < 1e-5)** | **Yes**\* |
| OSQP (ADMM) | 0.1057 | 283s | No (50K iter limit) | Yes\* |

\*Deterministic given same BLAS library

Scipy L-BFGS-B achieves **better loss** (0.1065 vs 0.1126) because it can properly optimize near bounds rather than stalling on clamping plateaus. Runtime of ~7 minutes is acceptable for a build step that runs once per release.

OSQP (a dedicated QP solver using ADMM) was also tested but could not converge to tight tolerances within reasonable time — a fundamental limitation of first-order ADMM methods for high-accuracy solutions.

## What Changes

- **Solver**: PyTorch L-BFGS → scipy L-BFGS-B (Fortran)
- **Gradient**: autograd → analytical (closed-form)
- **Bounds handling**: `torch.clamp()` → projected gradients (L-BFGS-B native)
- **Objective function**: unchanged
- **GPU dependency**: eliminated (CPU-only, deterministic)

## What Doesn't Change

- The objective function and its meaning
- The targets, penalties, and bound constraints
- The output format (weight multipliers applied to s006)

The branch `experiment-grad-norm` on the fork contains the working scipy implementation and all experimental logs.

### Part 2: Create gold-star weights and include them in master branch

In addition to the above, we can create a set of gold-star weights -- optimum weights produced on a contributor's machine. We can commit them to master. Users can check whether their weights match the gold-star weights within np.allclose() defaults, or they can just use the gold-star weights directly.
