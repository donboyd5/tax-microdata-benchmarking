**Read `repo_conventions_session_notes.md` first.**

# Optimization A+B Benchmark Results

*Date: 2026-03-24*
*Branch: `cd-pipeline`*

## What A+B Does

- **A:** Pre-cache DataFrame column arrays (c00100, MARS, data_source,
  variable columns) to avoid repeated `vardf[col].astype(float).values`
  in the per-target loop inside `_build_constraint_matrix()`.
- **B:** Build the sparse B matrix directly using COO format instead of
  assembling a dense `n_records x n_targets` intermediate (~0.3 GB for
  215K records x 107 targets) and then converting to sparse.

Both changes are in `create_area_weights.py` only.

## Benchmark Setup

- 49 CDs (every 9th area alphabetically), 16 workers, 107 targets
- Ryzen 9 9950X (16C/32T), WSL2
- Script: `benchmark_ab.py` — swaps baseline/optimized code and times
  both on the same subset

## Results

| Version | Wall time | Per area | Violations |
|---------|-----------|----------|-----------|
| Baseline (git HEAD) | 397.7s | 8.1s | 7 (0.50% max) |
| Optimized A+B | 388.2s | 7.9s | 7 (0.50% max) |
| **Speedup** | **+2.4%** | | **Identical** |

## Decision: Not Applied

The 2.4% speedup (~80s on a full 436-CD batch) does not justify the
added code complexity:
- COO assembly with row/col/data index arrays
- Variable cache dictionary
- Pre-computed scope masks

The bottleneck is Clarabel's interior-point iterations (~95% of solve
time), not matrix construction. The optimization was originally
designed for counties (3,143 areas with potentially more targets),
where the benefit would compound. Revisit if county weighting
performance becomes an issue.

The stash remains on `optimize-constraint-matrix` branch (`stash@{0}`)
for future reference.

## Optimization C: Relaxed Clarabel Tolerances

*Also tested 2026-03-24.*

Changed `tol_gap_abs`, `tol_gap_rel`, `tol_feas` from `1e-7` to `1e-5`.

| Version | Wall time | Per area | Violated areas | Violated targets |
|---------|-----------|----------|---------------|-----------------|
| Baseline (1e-7) | 397.7s | 8.1s | 7 | 7 |
| Relaxed (1e-5) | 388.4s | 7.9s | 19 | 92 |

**Decision: Not Applied.** Only 2.3% faster but 13x more violated
targets. The solver converges in fewer iterations but the looser
internal tolerance lets constraints slip past the 0.50% boundary.
Quality degradation far outweighs the negligible speed gain.

## Summary

Neither A+B nor C provides meaningful speedup for the CD pipeline.
The bottleneck is Clarabel's interior-point solver, which is already
well-optimized. For 436 CDs at 107 targets, ~55 min wall time with
16 workers is the practical floor without changing the solver or
problem formulation.
