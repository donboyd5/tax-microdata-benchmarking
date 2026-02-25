**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Constrained Optimization (NLP) Reweighting

*Branch: `experiment-nlp`*
*Last updated: 2026-02-25*

---

## Current Status

**Initial implementation complete and tested on real data (225K records, 548 targets).**
**IPOPT converged in 28 iterations, 820 seconds. All 548 constraints satisfied (zero slack).**

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

| Metric | PyTorch L-BFGS | IPOPT NLP |
|--------|---------------|-----------|
| Approach | Penalty-based | Constrained |
| Target guarantee | No (soft) | Yes (within ε) |
| Runtime | ~90s GPU, ~180s CPU | ~820s CPU |
| All within ±0.5% | ~81% | 81.9% (guaranteed) |
| Weight deviation | Unknown | 22.014 |
| Infeasibility handling | Manual target curation | Automatic (slack) |

**Key observation:** Runtime is slower than PyTorch L-BFGS (820s vs 90-180s).
The bottleneck is MUMPS linear algebra (808s of 820s total). HSL MA57/MA97
could significantly reduce this — the KKT system is sparse and structured.

### MUMPS tuning attempt
Tested with `OMP_NUM_THREADS=16`, `mumps_permuting_scaling=7`, `mumps_scaling=77`,
`nlp_scaling_method=gradient-based`. Result: 812.6s (vs 820s baseline) — negligible
improvement. The KKT structure (diagonal Hessian + sparse Jacobian) doesn't benefit
from MUMPS reordering/scaling. Real speedup requires HSL or Schur complement.

### Memory optimization
Fixed dense intermediate B matrix (225K × 548 × 8 bytes ≈ 1GB waste). Now uses
sparse-only path: `A_csc = csc_matrix(A); B_csc = spdiags(w0) @ A_csc`. Peak
memory for constraint matrix reduced from ~2GB to ~90MB.

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

### C. Dropped targets
- `unemployment_compensation` dropped initially (matching experiment-scipy-lbfgsb)
- 4 estate/rental variables already commented out in master
- Eventually: elastic formulation handles these automatically (no manual curation)

### D. Solver fallback
- Primary: IPOPT (cyipopt Problem class with analytical Hessian)
- Fallback: scipy trust-constr (no installation needed, uses BFGS approximation)

---

## Files

### New files
- `tmd/utils/reweight_nlp.py` — Constrained optimization solver
- `session_notes/nlp_reweighting_session_notes.md` — This file

### Modified files
- `tmd/imputation_assumptions.py` — NLP constants (tolerance, slack penalty, max iter)
- `tmd/datasets/tmd.py` — Switchable solver (NLP_REWEIGHT env var)
- `tmd/utils/reweight.py` — Drop UC from target list

### Reused from existing code
- `build_loss_matrix()` — builds output matrix and SOI targets
- `_drop_impossible_targets()` — drops all-zero columns
- Prescaling logic — match filer total to SOI target before optimization

---

## How to Resume

1. Switch to branch `experiment-nlp`
2. Read this file: `session_notes/nlp_reweighting_session_notes.md`
3. Key module: `tmd/utils/reweight_nlp.py`
4. Test: `NLP_REWEIGHT=1 make data` or run reweight_nlp() standalone

---

## Related Session Notes

- `reweighting_session_notes.md` — PR #407 (PyTorch L-BFGS penalty approach)
- `reweighting_experimentation_session_notes.md` — scipy L-BFGS-B experiments
