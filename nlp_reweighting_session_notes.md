**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Constrained Optimization (NLP) Reweighting

*Branch: `experiment-nlp`*
*Last updated: 2026-02-25 (evening ET)*

---

## Current Status

**Clarabel is default solver. Per-constraint tolerance overrides and dual-based
constraint cost reporting implemented. UC re-enabled with ±5% override.
Cross-machine reproducibility investigated — input data diverges slightly
between machines due to data generation pipeline, not solver.**

---

## First Run Results (2026-02-25)

| Metric | Value |
|--------|-------|
| Solver | IPOPT 3.11.9 (MUMPS linear solver) |
| Variables | 226,352 (225,256 multipliers + 1,096 slacks) |
| Constraints | 548 inequality (±0.5% tolerance) |
| Iterations | 28 |
| Wall time | 820 seconds (~14 min) |
| CPU in IPOPT | 808s (linear algebra), 3.4s (function evaluations) |
| Objective | 22.014 (sum of squared deviations from 1) |
| Constraint violations | 0 |
| Active slacks | 0 (all constraints satisfied) |

### Target accuracy
- 100% within ±1.0% (all 548 targets)
- 81.9% within ±0.5% (449/548)
- 107 constraints binding at ±0.5% boundary
- Worst targets are employment income counts/totals at ±0.500%

### Weight changes (vs pre-optimization)
- Median multiplier: 1.000000 (most weights unchanged)
- p5-p95 range: [0.995, 1.010]
- 87.4% of records changed by 0.1-1%
- Extreme range: [0.72, 1.59] — much tighter than allowed [0.1, 10]

### Comparison with PyTorch L-BFGS (PR #407)

| Metric | PyTorch L-BFGS | IPOPT NLP | Clarabel NLP |
|--------|---------------|-----------|-------------|
| Approach | Penalty-based | Constrained | Constrained |
| Target guarantee | No (soft) | Yes (within ε) | Yes (within ε) |
| Runtime | ~90s GPU, ~180s CPU | ~820s CPU | **~14s CPU** |
| All within ±0.5% | ~81% | 81.9% (guaranteed) | 98.2% (guaranteed) |
| Weight deviation | Unknown | 22.014 | 22.014 |
| Infeasibility handling | Manual target curation | Automatic (slack) | Automatic (slack) |
| Dependencies | PyTorch | cyipopt + MUMPS | clarabel (pip) |

**Clarabel is the recommended solver** — 57x faster than IPOPT, same solution
quality, and only requires `pip install clarabel` (no system libraries).

### Unemployment compensation experiments (2026-02-25)

UC was initially dropped from the target list because it's hard to match.
Three scenarios tested with Clarabel:

| Scenario | Targets | Objective | UC tolerance | Notes |
|----------|---------|-----------|-------------|-------|
| UC dropped | 548 | **22.01** | n/a | Baseline |
| UC at ±0.5% | 550 | **118** | 0.5% (default) | Severe weight distortion |
| UC at ±5% | 550 | **31.11** | 5% (relaxed) | Acceptable compromise |

UC at default ±0.5% causes 5.3x objective increase (22→118). Relaxing to ±5%
brings it down to 31 — still 41% above no-UC, but weight distortion is manageable.

Per-constraint tolerance overrides implemented:
```python
reweight_nlp(df, 2021, tolerance_overrides={"unemployment compensation": 0.05})
```

### Constraint cost reporting (2026-02-25)

Dual variables from Clarabel (`result.z`) now extracted and used to compute
marginal cost per 1 percentage point of tolerance relaxation:
`cost_per_pp[j] = dual[j] * |target_j| * 0.01`

Top 5 most expensive constraints (UC at ±5% run):

| Rank | cost/pp | Target |
|------|---------|--------|
| 1 | 26.40 | employment income/count/50k-75k |
| 2 | 25.19 | employment income/total/50k-75k |
| 3 | 9.57 | employment income/count/40k-50k |
| 4 | 9.22 | employment income/total/40k-50k |
| 5 | 8.09 | employment income/total/15k-20k |
| 11 | 4.66 | unemployment compensation/count (at ±5%) |

Interpretation: cost/pp is a marginal (local) estimate — the derivative of the
objective with respect to tolerance at the current solution. Employment income
constraints in the 40k-75k AGI range are the biggest drivers of weight distortion.
UC at ±5% still has significant marginal cost (4.66), suggesting it would need
±8-10% before its dual drops to near zero.

### Cross-machine reproducibility investigation (2026-02-25)

Compared Clarabel results on two machines, both running `make clean && make data`
from the same version of master. Found input data differs slightly **before**
optimization begins:

| Metric | Machine A (this) | Machine B (other) |
|--------|-------------------|-------------------|
| Target filers | 160,824,340 | 160,824,340 |
| Current filers (pre-scale) | 161,429,395 | 161,430,211 |
| Scale factor | 0.996252 | 0.996247 |
| **Filer difference** | — | **+816** |

Post-optimization results diverge as a downstream consequence:

| Metric | Machine A | Machine B |
|--------|-----------|-----------|
| Weight total | 183,488,178.91 | 183,491,268.25 |
| Weight mean | 814.576211 | 814.589925 |
| Weight sdev | 967.378494 | 967.487097 |
| Objective | **22.0140** | **21.8849** |
| sum(weights^2) | 360,264,434,413 | 360,316,801,114 |
| p50 | 574.040 | 574.301 |
| max | 16,919.445 | 16,918.580 |

**Root cause:** The data generation pipeline produces slightly different weights
on the two machines. The optimizer divergence (~0.6% objective difference) is too
large for solver non-determinism — it's driven by the 816-filer input difference.

**Likely causes for input divergence:**
- Different dependency versions (numpy, pandas, taxcalc)
- Non-deterministic operations in data generation (unseeded randomness, hash ordering)
- Platform floating-point differences (different CPU architecture / BLAS)

**Verification approach:** Copy generated data files from one machine to the other
and run only the optimization step. If results then match within solver tolerance
(~1e-8), the issue is confirmed as data-generation-only.

### MUMPS tuning attempt
Tested with `OMP_NUM_THREADS=16`, `mumps_permuting_scaling=7`, `mumps_scaling=77`,
`nlp_scaling_method=gradient-based`. Result: 812.6s (vs 820s baseline) — negligible
improvement. The KKT structure (diagonal Hessian + sparse Jacobian) doesn't benefit
from MUMPS reordering/scaling. Real speedup requires HSL or Schur complement.

### Memory optimization
Fixed dense intermediate B matrix (225K × 548 × 8 bytes ≈ 1GB waste). Now uses
sparse-only path: `A_csc = csc_matrix(A); B_csc = spdiags(w0) @ A_csc`. Peak
memory for constraint matrix reduced from ~2GB to ~90MB.

### Alternative solver results (2026-02-25)

| Solver | Time | Iterations | Objective | Status |
|--------|------|-----------|-----------|--------|
| IPOPT/MUMPS | 820s | 28 | 22.01392 | Optimal |
| OSQP (1K iter) | 6.2s | 1000 | 57.3 | Max iter (not converged) |
| OSQP (10K iter) | 55s | 10000 | 24.9 | Max iter (not converged) |
| **Clarabel** | **14.4s** | **33** | **22.01403** | **AlmostSolved** |

**Clarabel is the recommended solver.** 57x faster than IPOPT with identical
solution quality. Interior-point method (like IPOPT) but designed for sparse
conic QPs — no MUMPS dependency. Uses `faer` linear algebra (Rust, 32 threads).

OSQP (ADMM first-order method) converges quickly to rough solutions but slowly
to high precision. Not suitable for matching IPOPT's objective quality.

Weight distributions match between IPOPT and Clarabel to 4+ decimal places:
- IPOPT total: 183,488,183.32, objective: 22.0139152
- Clarabel total: 183,488,178.91, objective: 22.0140317

---

## Problem Statement

The current reweighting (PR #407, PyTorch L-BFGS) minimizes a combined objective:
target error + penalty for weight deviation. Targets are "soft" — the optimizer
tries to get close but doesn't guarantee hitting them. This experiment implements
the inverse formulation: **minimize weight deviation subject to target constraints**.

This is standard in survey calibration (calibration estimation) and gives:
- Clear separation: objective = minimize weight changes; constraints = hit targets
- Guaranteed target satisfaction (within tolerance) when feasible
- Automatic identification of infeasible constraints via elastic/slack variables
- No penalty-parameter tuning (replaced by constraint tolerance)

---

## Formulation

### Core problem

```
minimize    sum((x_i - 1)^2)                               [weight deviation]
subject to  t_j - |t_j|*ε ≤ (B^T x)_j ≤ t_j + |t_j|*ε    [target constraints]
            0.1 ≤ x_i ≤ 10.0                               [multiplier bounds]

where:
  x_i   = weight multiplier for record i  (N ≈ 225K)
  B_ij  = w0_i * A_ij  (prescaled weight × output matrix value)
  t_j   = SOI target value
  ε     = constraint tolerance (default 0.5%)
```

### Elastic extension (for graceful infeasibility handling)

```
minimize    sum((x_i - 1)^2) + M * sum(s_lo_j^2 + s_hi_j^2)
subject to  t_j - |t_j|*ε ≤ (B^T x)_j + s_lo_j - s_hi_j ≤ t_j + |t_j|*ε
            0.1 ≤ x_i ≤ 10.0,  s_lo_j ≥ 0,  s_hi_j ≥ 0

Variables: y = [x (N); s_lo (M); s_hi (M)]
```

- Hessian is diagonal and constant: `diag([2,...,2, 2M,...,2M, 2M,...,2M])`
- Constraints are linear → this is a convex QP
- IPOPT solves in ~6-10 iterations (verified in smoke tests)

### Why this is different from the Lagrangian/penalty approach

An augmented Lagrangian collapses back to `penalty * constraint_violation^2` in the
objective — essentially what the current code already does. The constrained approach
is structurally different: IPOPT uses an interior-point method with barrier functions,
solving the KKT system directly. Constraints are first-class, not penalties.

The elastic/slack formulation is also different from putting targets in the objective:
slacks only activate when mathematically necessary, and their values tell you exactly
which constraints are problematic and by how much.

---

## Solver: IPOPT via cyipopt

**Why IPOPT:**
- Interior-point method (not penalty-based)
- Handles bounds + inequality/equality constraints natively
- Exploits sparse structure (diagonal Hessian, ~2% dense Jacobian)
- Built-in infeasibility detection
- 225K variables is well within IPOPT's comfort zone

**cyipopt version:** 1.6.1 (installed)

**HSL subroutines:** User has license but not installed. Default MUMPS linear solver
should be sufficient. HSL MA57/MA97 available as future optimization via
`options={'linear_solver': 'ma57'}`.

**Smoke test results (planning phase):**
- 10K vars, 50 constraints: 9 iterations, 0.09s
- 50K vars, 200 constraints: 6 iterations, 7.2s
- Full problem (225K vars, 550 constraints): estimated 30-120s

**Actual results (full problem):**
- 225K vars, 548 constraints: 28 iterations, 820s
- Bottleneck: MUMPS factorization (808s of 820s)
- Function evaluations only 3.4s — problem structure is efficient
- HSL subroutines would likely cut runtime significantly

---

## Approach: Key Decisions

### A. Objective function
- Starting with `sum((x_i - 1)^2)` (chi-squared calibration distance)
- Framework extensible to Cressie-Read family `sum(x^4 + x^(-4) - 2)` later
- Non-quadratic objectives lose QP structure but IPOPT handles general NLP

### B. Constraints
- Inequality with ±0.5% tolerance (configurable)
- For near-zero targets: minimum absolute tolerance (100) to avoid ±0% = equality
- Elastic/slack formulation always active with large penalty (1e6)
- Non-zero slacks identify problematic constraints in output

### C. Target handling
- `unemployment_compensation` re-enabled with ±5% tolerance override
- 4 estate/rental variables still commented out in master
- Per-constraint tolerance overrides handle problematic targets without dropping them
- Dual-based constraint cost reporting identifies expensive constraints automatically

### D. Solver
- Primary: Clarabel (default) — 14s, pip-installable, interior-point conic QP
- Alternative: IPOPT (820s, requires cyipopt + MUMPS)
- Fallback: scipy trust-constr (no installation needed, uses BFGS approximation)

---

## Files

### New files
- `tmd/utils/reweight_nlp.py` — Constrained optimization solver (Clarabel/IPOPT)
- `test_nlp_reweight.py` — Standalone test script (runs Clarabel on existing tmd.csv.gz)
- `session_notes/nlp_reweighting_session_notes.md` — This file

### Modified files
- `tmd/imputation_assumptions.py` — NLP constants (tolerance, slack penalty, max iter)
- `tmd/datasets/tmd.py` — Switchable solver (NLP_REWEIGHT env var), Clarabel default
- `tmd/utils/reweight.py` — UC re-enabled (handled via tolerance override)

### Reused from existing code
- `build_loss_matrix()` — builds output matrix and SOI targets
- `_drop_impossible_targets()` — drops all-zero columns
- Prescaling logic — match filer total to SOI target before optimization

---

## How to Resume

1. Switch to branch `experiment-nlp`
2. Read this file: `session_notes/nlp_reweighting_session_notes.md`
3. Key module: `tmd/utils/reweight_nlp.py`
4. Quick test: `python test_nlp_reweight.py` (uses existing tmd.csv.gz)
5. Full pipeline: `NLP_REWEIGHT=1 make data`
6. Install: `pip install clarabel` (only new dependency)

---

## Related Session Notes

- `reweighting_session_notes.md` — PR #407 (PyTorch L-BFGS penalty approach)
- `reweighting_experimentation_session_notes.md` — scipy L-BFGS-B experiments
