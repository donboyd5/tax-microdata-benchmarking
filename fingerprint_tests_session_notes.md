# Fingerprint Tests for Area Weights — Session Notes

## Date: 2025-03-30

## Context

Reviewed a Martin-Claude conversation that explored weaknesses in the current
area weight fingerprint tests and alternatives. Don then asked for deeper
analysis of cross-machine reproducibility concerns.

## Current approach (from Martin-Claude conversation)

- Per area: round each weight to nearest integer, sum → one integer per area
- Hash: sort areas, join as `AK:12345|AL:67890|...`, SHA-256 (first 16 chars)
- Tests: hash comparison + per-area integer sum comparison

## Problems identified

1. **Insensitive to small areas**: weights often 0.1–8.0; rounding to integer
   destroys information (e.g., 1.5 → 2.4 is a 60% change, both round to 2)
2. **Distribution-blind**: different weight vectors with same integer sum pass
3. **Rounding boundary false positives**: 2.4999 vs 2.5001 (numerical noise)
   rounds to 2 vs 3, causing a spurious failure
4. **Hash provides no diagnostics**: only tells you something changed, not what

## Alternatives analyzed

- **Full-vector hash (full precision)**: too sensitive, fails across machines
- **Full-vector hash (rounded precision)**: rounding boundary problem with 215K weights
- **Relative tolerance on sums only**: fixes rounding issues but still distribution-blind
- **Multiple statistics + rtol**: recommended — no boundary problems, distribution-sensitive, diagnostic

## Recommendation

Five statistics per area on `WT{TAXYEAR}` column:

| Statistic | Match type | Catches |
|-----------|-----------|---------|
| `n_records` | Exact | Data pipeline changes |
| `weight_sum` | rtol=1e-3 | Level shifts (1st moment) |
| `weight_std` | rtol=1e-3 | Distribution shape (2nd moment) |
| `n_positive` (>0.01) | Exact | Sparsity pattern changes |
| `max_weight` | rtol=1e-3 | Tail/outlier behavior |

**Tolerance rationale**: Cross-machine noise is O(1e-6 to 1e-4) relative;
real changes are typically >1%. rtol=1e-3 sits cleanly between with margin
on both sides. With 3,000+ areas (future counties), false positive probability
remains negligible.

**Key insight**: This builds on the existing `test_weights.py` pattern
(multi-statistic with `np.allclose`) and on Martin-Claude option 3
(relative tolerance on sums), adding `weight_std` and `max_weight`
to close the distribution-blind gap.

## Status

- GitHub issue opened by Don
- Awaiting initial feedback from Martin before implementation
- Will return to this after Martin's review
