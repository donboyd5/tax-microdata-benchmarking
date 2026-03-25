# Clarabel Solver Reuse Benchmark (2026-03-24)

## Idea

Clarabel supports `solver.update()` to change problem data (P, q, A, b)
without rebuilding the solver, reusing symbolic factorization.  All CD
areas share the same constraint matrix sparsity pattern (same national
microdata, same target spec structure), so only `pop_share` and `targets`
change between areas.

Source: https://clarabel.org/stable/user_guide_data_updating/

## Constraint

`solver.update()` requires `presolve_enable = False` AND
`chordal_decomposition_enable = False`.  Both are `True` by default
in Clarabel, and our production code uses the defaults.

## Results

Tested on 18 CDs (20 selected, 2 skipped due to dropped targets
changing the matrix shape), 107 targets each.

**With presolve/chordal disabled (required for reuse):**
All 18 areas return `DualInfeasible` with 107/107 violations for
BOTH fresh and reuse methods.  The solver cannot converge without
these numerical conditioning features.

**Speedup measurement (invalid solutions, but timing is meaningful):**
- Fresh: 226.0s (12.6s/area)
- Reuse: 206.0s (11.4s/area)
- Speedup: +8.8%

The ~9% speedup from avoiding solver reconstruction is real, but
unusable because the required settings prevent convergence.

## Conclusion

Solver reuse via `update()` is **not viable** with Clarabel for our
problem.  The presolve and chordal decomposition features do critical
numerical conditioning for the 215K-variable, 107-constraint QP with
elastic slack.  Without them, the solver reports dual infeasibility.

The ~30s per area with default Clarabel settings is effectively the
floor for this problem size.  Future speedup paths:
- Parallelism (already at 16 workers, ~55 min wall time for 436 CDs)
- Fewer targets (reduces QP size but sacrifices accuracy)
- Future Clarabel releases may relax the update() restrictions
