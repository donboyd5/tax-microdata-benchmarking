# EITC and CTC Per-Bin Targeting: What We Learned

Analysis from CD pipeline development, 2026-03-24.

## Summary

We initially believed per-bin CTC targets were fundamentally
incompatible with per-bin EITC targets due to record-level credit
entanglement.  **This was wrong.**  The real culprit was a duplicate
shares bug in `_add_ctc_total()` that created two conflicting CTC
share rows per area (one for nonrefundable CTC only, one for the
correct CTC + ACTC total).  After fixing the bug, the 107-target
spec (with both EITC and CTC per-bin) solves cleanly across all
436 congressional districts.

## The Bug

`_add_ctc_total()` in `prepare_shares.py` combines SOI columns
07225 (nonrefundable CTC/ODC) and 11070 (refundable ACTC) into a
combined `ctc_total` row.  But it did not remove the original 07225
rows, leaving two conflicting share entries for every `ctc_total`
target across all 436 CDs (8,720 duplicate groups in the shares
file).  The solver saw contradictory constraints -- one using the
nonrefundable-only share, one using the correct combined share --
and thrashed.

**Fix:** Remove original 07225 and 11070 rows before appending the
combined rows.  Shares file dropped from 126,876 to 118,156 rows,
0 duplicates.

## Before and After

### Three-CD test (single area solves)

| Area | 107 tgts, BUGGY shares | 107 tgts, FIXED shares |
|------|----------------------|----------------------|
| NY12 | 35s, 38 viol, 0.50%  | 38s, 3 viol, 0.50%   |
| FL28 | **104s, 19 viol, 15.52%** | 31s, 1 viol, 0.50% |
| TX20 | **93s, 46 viol, 32.23%**  | 31s, 0 viol         |

### Full 436-CD batch (107 targets, 16 workers)

| Metric | Previous best (95 tgts, buggy) | New (107 tgts, fixed) |
|--------|-------------------------------|----------------------|
| Targets per CD | 95 | **107** |
| Failed areas | 0 | 0 |
| Areas w/ violations | 49 | 55 |
| Total violated targets | 88 | **69** |
| Largest violation | 0.50% | 0.50% |
| Wall time | ~54 min | ~55 min |
| Returns aggregation | — | -0.87% |
| AGI aggregation | — | -0.02% |

More targets, fewer total violations, similar wall time.

## Why We Were Misled

The initial diagnosis looked compelling:

1. **Real geographic divergence:** EITC and CTC shares genuinely
   differ -- EITC is 91% concentrated in $10K-$50K AGI, while CTC
   spreads across $10K-$500K.  Low-income CDs need +60% EITC but
   only average CTC in the same bins.

2. **Record-level co-occurrence:** 97.5% of EITC recipients in
   $25K-$50K also have CTC.  Under tax law (IRC 24 and 32), a child
   under 17 qualifies for both credits.  Only children age 17-18
   qualify for EITC but not CTC.

3. **Timing matched:** Adding CTC per-bin targets caused solve time
   explosions (12s → 92s on AL01 in earlier testing), and EITC
   per-bin targets did not.

This all pointed to "fundamental EITC/CTC entanglement."  But the
actual cause was simpler: the solver was fighting contradictory CTC
share constraints, not EITC-vs-CTC conflicts.  The EITC targets were
cheap because EITC has no similar duplicate-share issue (it uses a
single SOI column, 59660).

## What Remains True

The underlying facts about EITC/CTC are still correct and worth
knowing for future recipe design:

- **Co-occurrence is real:** 97.5% at $25K-$50K, driven by tax law
  (qualifying children trigger both credits).
- **Geographic divergence is real:** Cross-CD correlation of EITC vs
  CTC shares is only 0.27.  EITC-heavy CDs tend to have lower CTC
  shares in the same bins.
- **CTC composition shifts:** At low AGI, refundable ACTC dominates
  (94% at $10K-$25K); at mid-AGI, nonrefundable CTC rises to 80%
  at $50K-$75K.  This drives the geographic divergence.
- **SOI data quality is good:** Both EITC (A59660) and CTC (A07225 +
  A11070) come directly from tax returns.  No data quality concerns.

These factors make EITC/CTC per-bin targeting harder than most
variable pairs, but with correct shares the solver handles it fine.
The 107-target spec runs in similar time to the 95-target spec.

## Lessons

1. **Always check for duplicate shares** when combining multi-source
   variables.  Added `test_no_duplicate_cd_shares` to the test suite.

2. **Solver explosions usually have a data cause**, not a fundamental
   optimization limit.  Before concluding "the solver can't handle
   this," check the inputs.

3. **The initial "fundamental conflict" diagnosis was plausible but
   wrong.**  Real geographic divergence + real record-level
   entanglement + timing evidence all pointed to an inherent limit.
   But the simpler explanation (bad data) was the actual cause.

## SOI Data Note: A59664 Unit Error

Separately, we found that SOI column A59664 (EITC amount for 3+
qualifying children) is published in dollars rather than $1,000s in
the 2022 CD file.  The state file is correct.  This does not affect
our pipeline (we only use A59660, the total), but a workaround was
added to `soi_cd_data.py` and an email drafted for SOI in
`session_notes/soi_a59664_unit_error_email.md`.
