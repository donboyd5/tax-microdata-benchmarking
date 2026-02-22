**Read `repo_conventions_session_notes.md` first.**

# GPU vs CPU Reweighting Reproducibility Investigation

## Branch: experiment-grad-norm

## Problem Statement

GPU and CPU reweighting produce different weights (`np.allclose()` fails).
Root cause: PyTorch L-BFGS uses `torch.clamp()` for bounds, creating flat
gradient regions. On the flat loss surface, floating-point non-associativity
in `.sum()` operations causes different reduction orderings on GPU vs CPU,
pushing records to opposite bound extremes (0.1 on GPU, 10.0 on CPU).

This is NOT specific to GPU vs CPU -- any two different compute backends
(different CPUs, different BLAS libraries) will diverge.

## Approaches Tested

### 1. Tighten grad_norm_tol (PyTorch L-BFGS)
- Tried 1e-5, 5e-6, 1e-6, 1e-7, 1e-8
- Result: **FAILED** -- never achieved `np.allclose()`. Tighter tolerance
  doesn't help because the opposite-bound problem is structural (clamping).

### 2. Tighter bounds [0.5, 2.0] (PyTorch L-BFGS)
- Result: Loss too high (4.17 vs 0.11 with default bounds)

### 3. Pure L-BFGS-B optimizer (nlesc-dirac port to PyTorch)
- Created `tmd/utils/lbfgsb.py` -- vectorized Python L-BFGS-B
- Result: **TOO SLOW** -- Cauchy point loop iterates over all 225K params
  sequentially in Python. Run killed after producing no output.

### 4. Scipy L-BFGS-B (Fortran implementation) **<-- RECOMMENDED**
- Proper projected-gradient bounds (no clamping)
- Same objective function as PyTorch version
- Analytical gradient (not autograd)
- Result: **CONVERGED** -- 2940 iterations, 422 seconds, loss=0.1065
- Better loss than PyTorch GPU (0.1126) due to proper bounds handling
- Deterministic across CPU machines with same BLAS
- The QP has a **unique global minimum** (P is positive definite), so any
  solver that converges tightly enough will produce the same answer

### 5. OSQP (ADMM-based QP solver)
- Same objective reformulated as standard QP with auxiliary variables
- Reformulation avoids forming dense 225K x 225K P matrix
- Result: **DID NOT CONVERGE** at eps=1e-5 after 50K iterations (~280s)
- Solution quality similar (loss=0.1057) but duality gap closure is slow
- Tried rho/sigma tuning -- marginal improvement, fundamental ADMM limitation
- First-order method, converges slowly to high accuracy

## Recommendation

**Use scipy L-BFGS-B as the production solver.** Rationale:

1. **Correctness**: Proper projected-gradient bounds (no clamping artifacts)
2. **Uniqueness**: QP has unique minimum -- any machine converging tightly
   enough gets the same answer
3. **Determinism**: Fortran L-BFGS-B is deterministic across CPU machines
   (same BLAS). Not GPU-dependent.
4. **Quality**: Achieves better loss (0.1065) than PyTorch GPU (0.1126)
5. **Runtime**: 422 seconds (~7 min) -- acceptable for a build step that
   runs once per release

### Implementation strategy (gold-star weights)

Option A: Run scipy L-BFGS-B as the production solver in `make tmd_files`
  - Replace PyTorch L-BFGS entirely
  - ~7 min runtime, no GPU needed
  - Cross-machine reproducible

Option B: Generate reference weights once, commit them
  - Run scipy L-BFGS-B once to produce "gold star" weights
  - Commit weights to repo
  - Other machines verify against committed weights
  - Faster builds (skip reweighting), guaranteed reproducibility

## Key Results Summary

| Solver | Loss | Time | Converged | Cross-machine |
|--------|------|------|-----------|---------------|
| PyTorch GPU L-BFGS | 0.1126 | ~90s | Yes (stagnated) | No |
| PyTorch CPU L-BFGS | 0.1126 | ~180s | Yes (stagnated) | No |
| Scipy L-BFGS-B | 0.1065 | 422s | Yes (grad<1e-5) | Yes* |
| OSQP | 0.1057 | 283s | No (50K iter limit) | Yes* |

*Deterministic given same BLAS library

## Files in This Branch

### New files:
- `compare_gpu_cpu.py` -- Script to run GPU vs CPU comparisons
- `run_single_reweight.py` -- Single reweight runner (subprocess isolation)
- `tmd/utils/lbfgsb.py` -- L-BFGS-B optimizer (Python port, too slow)

### Modified files:
- `tmd/utils/reweight.py` -- Added `use_scipy`, `use_osqp`, `use_lbfgsb`
  parameters with full implementations
- `tmd/datasets/tmd.py` -- Saves pre-reweight snapshot for experiments
- `tmd/imputation_assumptions.py` -- Added REWEIGHT_GRAD_NORM_TOL constant
- `run_single_reweight.py` -- CLI flags for all solver options

### Log files (in compare/):
- Various GPU/CPU/scipy/OSQP logs from experiments
- `pre_reweight_snapshot.csv.gz` -- 225K record starting point (62MB, gitignored)

## Objective Function (all solvers use the same formulation)

```
loss = sum_j [ (sum_i w0_i * m_i * A_ij) / (t_j + 1) - t_j / (t_j + 1) ]^2
     + penalty * L0 * sum_i (w0_i^2 / sum(w0^2)) * (m_i - 1)^2

subject to: multiplier_min <= m_i <= multiplier_max
```

Where:
- m_i = weight multiplier for record i (decision variable)
- w0_i = original (prescaled) weight for record i
- A_ij = output matrix (record i contribution to target j)
- t_j = SOI target value for target j
- L0 = initial unpenalized loss (for scaling the penalty)
- penalty = REWEIGHT_DEVIATION_PENALTY (0.01)

The scipy implementation uses analytical gradient:
```
grad_i = 2 * sum_j B_ij * (sum_k B_kj * m_k - c_j) + 2 * lam_i * (m_i - 1)
```
where B_ij = w0_i * A_ij / (t_j + 1), c_j = t_j / (t_j + 1)
