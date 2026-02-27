**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Constrained Optimization (NLP) Reweighting

*Branch: `experiment-nlp`*
*Last updated: 2026-02-27 (morning ET)*

---

## Current Status

**Clarabel is default solver (Clarabel replaces PyTorch L-BFGS as default in
`tmd.py`). Constraint scaling enabled by default. Strict ±0.5% tolerance on
all 548 constraints (no absolute floor). Slack penalty increased to 1e7.
UC targets removed (2021 anomalies). Cross-machine reproducibility CONFIRMED.**

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

**Machine environment comparison (2026-02-26):**

| Item | business-new (Machine A) | business-old (Machine B) |
|------|--------------------------|--------------------------|
| Python | 3.12.3 | 3.12.3 |
| CPU | Ryzen 9 9950X (Zen 5, AVX-512) | FX-8350 (Piledriver, AVX only) |
| Platform | WSL2 | Native Linux |
| numpy | 2.1.3 | 2.1.3 |
| **pandas** | **2.3.3** | **2.3.2** |
| **scipy** | **1.16.2** | **1.16.1** |
| taxcalc | 6.4.0 | 6.4.0 |
| **tax-calculator** | **1.2.4** | **(not installed)** |
| **policyengine-core** | **3.20.1** | **3.20.0** |
| policyengine-us | 1.55.0 | 1.55.0 |
| **numba** | **0.62.1** | **0.61.2** |
| clarabel | 0.11.1 | 0.11.1 |

**Top suspects for 816-filer input divergence:**
1. **policyengine-core 3.20.1 vs 3.20.0** — tax policy engine; even patch versions
   can change computation results affecting filer status or imputed values
2. **tax-calculator 1.2.4** present on new, absent on old — if any code path
   imports from it, behavior would differ
3. **CPU FP differences** — Piledriver (FX-8350, no AVX2/512) vs Zen 5 (9950X,
   AVX-512) have different FPU implementations; NumPy/BLAS uses different
   instruction paths, and threshold-sensitive records could flip filer status

**Recommended fix:** Pin policyengine-core to same version on both machines,
ensure tax-calculator presence matches, re-run `make clean && make data`.

**Verification approach:** Copy generated data files from one machine to the other
and run only the optimization step. If results then match within solver tolerance
(~1e-8), the issue is confirmed as data-generation-only.

### Code analysis: non-determinism in data generation (2026-02-26)

Thorough review of the `make data` pipeline identified the probable root cause
and secondary risk factors.

**Candidate root causes — binary thresholds in data pipeline:**

Two places in `tmd/datasets/tmd.py` apply boolean filters that determine the
record set:

1. **Line 28:** `tax_unit_is_filer` — PolicyEngine Microsimulation classifies CPS
   records as filers. *Note: this is likely a boolean/categorical variable, not
   a continuous FP value, so FP threshold sensitivity may not apply here.
   Needs investigation of PolicyEngine internals to confirm.*
2. **Line 42:** `iitax > 0` — Tax-Calculator computes income tax for CPS records;
   those with positive tax are dropped. Borderline FP values near zero could
   resolve differently on different hardware.

The 816 difference is in **weighted filers** (sum of s006), not necessarily 816
individual records. A small number of records with weights summing to 816 could
account for the difference.

**Secondary risk factors identified:**

| Risk | Source | Location |
|------|--------|----------|
| HIGH | TaxCalc `iitax > 0` FP threshold; PE filer TBD | `tmd.py:28,42` |
| MED-HIGH | `taxcalc>=6.4.0` not strictly pinned | `setup.py:12` |
| MEDIUM | scikit-learn not pinned (tree imputation) | `setup.py:13` |
| MEDIUM | `astype(int)` truncation after FP diffs | `create_taxcalc_input_variables.py:55` |
| MEDIUM | `set()` iteration in MICE (no PYTHONHASHSEED) | `mice.py:447,452,509` |
| LOW-MED | CPU-specific BLAS paths (AVX vs AVX-512) | numpy/scipy/torch throughout |

**What's NOT a problem:** All random number generators are properly seeded
(cps.py, puf.py, imputation.py, reweight.py, mice.py). RNG usage is deterministic
*given identical inputs*. The non-determinism enters upstream via FP differences
that change which data flows through the seeded RNG code.

**Probable causal chain:**
1. Different CPU/BLAS produces slightly different FP intermediates in PolicyEngine
   and/or Tax-Calculator
2. Borderline tax units flip at `iitax > 0` (`tmd.py:42`) and/or filer
   classification (`tmd.py:28`, mechanism TBD) — accounts for 816 weighted filers
3. Different record set enters reweighting, producing different optimal weights
4. Weighted totals diverge (~3K difference in weight total, ~0.6% in objective)

### Cross-machine reproducibility RESOLVED (2026-02-26 evening)

**Root cause found:** The previous cross-machine comparison was flawed. `test_nlp_reweight.py` loaded `tmd.csv.gz` (which contains POST-PyTorch-reweighting weights) and fed those to Clarabel. Since each machine's PyTorch run produced slightly different weights (due to the 816-filer input difference), Clarabel received different starting points on each machine — explaining the apparent divergence.

**Fix:** Modified `tmd/datasets/tmd.py` to save pre-reweighting weights to `tmd/storage/output/pre_reweight_weights.csv.gz` before the reweighting subprocess runs. Then ran `NLP_REWEIGHT=1 make clean && NLP_REWEIGHT=1 make data` on both machines.

**Result: Both machines produce IDENTICAL output:**

| Metric | Machine A | Machine B |
|--------|-----------|-----------|
| Pre-scaled filers | 161,180,573 | 161,180,573 |
| Scale factor | 0.997790 | 0.997790 |
| sum(weights^2) | 400,090,259,974.648621 | 400,090,259,974.648621 |
| Objective (sum((x-1)^2)) | 105,285.2561 | 105,285.2561 |
| Clarabel iterations | 33 | 33 |
| Solve time | ~14s | ~14s |

The previous divergence was entirely an artifact of the test script using wrong starting weights. The Clarabel solver is perfectly deterministic across machines when given the same input.

**Note:** The pre-reweighting s006 values only exist transiently during `make data` (in a temp directory). They are now also saved to `pre_reweight_weights.csv.gz` for debugging. `tmd.csv.gz` and `tmd_weights.csv.gz` both contain POST-reweighting weights.

---

### QP solver exploration via `qpsolvers` (2026-02-27)

Exploring alternative QP solvers for a related project requiring hundreds of similar solves. Created `sandbox_qpsolvers.py` to benchmark solvers on a synthetic problem matching TMD dimensions (225K variables, 550 constraints, 2% density).

**Problem formulation (same as reweight_nlp.py):**
```
minimize    sum((x_i - 1)^2) + lambda * penalty(s)
subject to  G @ x - s <= h    (target bounds, stacked upper+lower = 1100 rows)
            x in [0.1, 10.0], s >= 0
```

Two penalty modes: L1 (lambda * sum(s)) and L2 (lambda * sum(s^2)).

**Results WITHOUT constraint scaling (lambda=1e6):**

| Solver | Type | Status | Time | Notes |
|--------|------|--------|------|-------|
| **Clarabel** | Interior-point | OK | ~44s | 550/550 within tol, 0 slacks |
| OSQP | First-order (ADMM) | FAILED | — | Diverges, can't handle 1e6 penalty |
| PIQP | First-order | FAILED | — | Diverges similarly |
| ProxQP | Interior-point | HUNG | — | Didn't start solving, killed |
| HiGHS | Active-set | SLOW | >60s | Converging but very slowly; warned about "excessively large row bounds" |

**Key insight from HiGHS diagnostics:**
```
Coefficient ranges:
  Cost    [2e+00, 1e+06]    ← 500,000:1 ratio (lambda in cost vector)
  RHS     [1e+08, 1e+08]    ← huge constraint bounds (aggregate targets)
WARNING: Problem has some excessively large row bounds
WARNING:    Consider scaling the bounds by 1e-3
```

**Constraint scaling implementation:** Divide each constraint row by |target_j|:
- RHS normalises from ~1e8 to ~1.0 (the ±0.5% tolerance band)
- Slacks now measure *relative* constraint violation
- Lambda can be reduced (1e4 instead of 1e6) since slack units changed

**Results WITH constraint scaling (lambda=1e4):**

| Solver | Status | Time | Targets met | Weight dev | Notes |
|--------|--------|------|-------------|------------|-------|
| **Clarabel** | OK | 44s | 550/550 (100%) | 2904.90 | Same quality; already does internal equilibration |
| OSQP | Max iter (4K) | 49s | — | — | **Converging now** (vs diverging before), needs more iterations |
| HiGHS | Converging | >35s | — | — | No more warnings, objective declining steadily |
| PIQP | FAILED | — | — | — | Still struggles even with scaling |

**Coefficient ranges after scaling:**
```
  Matrix  [7.5e-07, 1.0e+00]
  Cost    [2.0e+00, 1.0e+04]    ← 5000:1 ratio (100x better)
  RHS     [9.9e-01, 1.0e+00]    ← normalised (8 orders of magnitude better)
  Bound   [1.0e-01, 1.0e+01]
```

**Conclusions so far:**
1. Constraint scaling is essential for first-order solvers
2. Clarabel remains the most reliable choice (interior-point, handles bad conditioning internally)
3. OSQP becomes viable with scaling + more iterations — potentially faster for "good enough" solutions
4. Should apply scaling to the real problem in `reweight_nlp.py` if switching to OSQP or other first-order solvers

**File:** `sandbox_qpsolvers.py` — standalone benchmark script with `scale_constraints` parameter, `**solver_kwargs` passthrough, coefficient range diagnostics.

---

### Constraint scaling applied to production solver (2026-02-27)

Constraint scaling (from `sandbox_qpsolvers.py` experiments) implemented in
`_solve_clarabel` in `reweight_nlp.py`. Each constraint row is divided by
|target_j|, normalizing RHS from ~1e8 to ~1.0. Dual variables are converted
back to original space after solving.

**Enabled by default** — `targets=target_array` is always passed to
`_solve_clarabel`. Can be disabled with `CLARABEL_NO_SCALE=1` env var.

### Strict ±0.5% tolerance — absolute floor removed (2026-02-27)

`_ABS_TOL_FLOOR` changed from 100.0 to 0.0. Previously, near-zero targets
(e.g., count=1125) got effective tolerance of ±8.9% because
`max(1125*0.005, 100) = 100`. With constraint scaling normalizing all
constraints, the generous floor became even more visible to the solver.

Now all 548 constraints use strict ±0.5% relative tolerance. Elastic slack
handles any that are mathematically infeasible.

### Slack penalty increased to 1e7 (2026-02-27)

`NLP_SLACK_PENALTY` in `imputation_assumptions.py` changed from 1e6 to 1e7.
Higher penalty pushes harder on constraint satisfaction at the cost of slightly
more weight distortion for infeasible constraints.

### UC targets removed (2026-02-27)

Unemployment compensation targets (2 constraints: count + total) commented out
in `reweight.py` `aggregate_level_targeted_variables`. 2021 UC anomalies made
these targets unreliable. Constraint set reduced from 550 to 548.

### Floating-point reporting fix (2026-02-27)

Added `+ 1e-9` epsilon to target accuracy bin comparisons in `_print_diagnostics`.
Solver satisfies constraints to `tol_feas=1e-7`, so binding constraints at ±0.5%
could land at 0.0050000001, causing `<= 0.005` to spuriously fail. The epsilon
prevents this noise from affecting the reported counts.

---

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
- Inequality with ±0.5% strict relative tolerance (configurable)
- No absolute tolerance floor (`_ABS_TOL_FLOOR = 0.0`); elastic slack handles infeasibility
- Elastic/slack formulation always active with penalty M=1e7 (`NLP_SLACK_PENALTY`)
- Non-zero slacks identify problematic constraints in output
- Constraint scaling (divide by |target_j|) enabled by default for Clarabel

### C. Target handling
- `unemployment_compensation` removed (2021 anomalies; 2 constraints dropped)
- 4 estate/rental variables still commented out in master
- Per-constraint tolerance overrides available for problematic targets
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
- `sandbox_qpsolvers.py` — QP solver benchmark (qpsolvers package, constraint scaling)
- `session_notes/nlp_reweighting_session_notes.md` — This file

### Modified files
- `tmd/imputation_assumptions.py` — NLP constants (tolerance, slack penalty=1e7, max iter)
- `tmd/datasets/tmd.py` — Clarabel default (PYTORCH_REWEIGHT=1 for PyTorch), saves pre-reweight weights
- `tmd/utils/reweight.py` — UC targets removed (2021 anomalies)

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
5. Full pipeline: `make clean && make data` (Clarabel is default)
6. To use PyTorch instead: `PYTORCH_REWEIGHT=1 make data`
7. To disable constraint scaling: `CLARABEL_NO_SCALE=1 make data`
8. Install: `pip install clarabel` (only new dependency)

---

## Related Session Notes

- `reweighting_session_notes.md` — PR #407 (PyTorch L-BFGS penalty approach)
- `reweighting_experimentation_session_notes.md` — scipy L-BFGS-B experiments
