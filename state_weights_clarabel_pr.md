# State Weights Clarabel PR

## Session Notes — 2026-03-18

### Branch: `state-weights-clarabel`

### Commits (12)
1. `bb56d06` — Add Clarabel QP solver and share-based state targeting pipeline
2. `2333b5a` — Use separate target/weight subfolders for states and CDs
3. `2c87ea6` — Exclude PR and US from state target file generation
4. `c2b7afe` — Change default SOI area data year from 2021 to 2022
5. `7b65f57` — Add cross-state quality summary report
6. `bfd320a` — Print SOI year in prepare and solve stage messages
7. `a9da9a6` — Optimize quality report: load data once, read only needed columns
8. `ff9241d` — Switch to full recipe, update variable mapping to shared names
9. `1883c11` — Exclude e26270 and SALT nz-counts from recipe for solver stability
10. `f7ba80e` — Add largest violation percentage to solve summary output
11. `8581cf1` — Restore e26270 to recipe (SOI discrepancy was analysis bug)
12. `aba0cd7` — Replace e18400/e18500 with c18300 (actual SALT) targeting

### PR Strategy (revised 2026-03-18)
- **Two PRs to upstream** (mirroring national reweighting approach):
  - PR 1: Target preparation pipeline (prepare/ modules, recipe, mapping CSV)
  - PR 2: Solver + quality report (builds on PR 1's target files)
- **Two branches**: Branch 1 = target prep only, Branch 2 = solver on top
- User will run the actual push/PR commands

### Current Recipe (12 combos, 118 targets/state)
| Variable | Type | SOI Base | Description |
|----------|------|----------|-------------|
| c00100 | amount | 00100 | AGI |
| c00100 | allcount | n1 | Total returns |
| c00100 | allcount | mars1 | Single returns |
| c00100 | allcount | mars2 | MFJ returns |
| c00100 | allcount | mars4 | HoH returns |
| e00200 | amount | 00200 | Wages |
| e00200 | nz-count | 00200 | Wages nonzero count |
| e00300 | amount | 00300 | Taxable interest |
| e01500 | amount | 01700 | Pensions (shared by taxable) |
| e02400 | amount | 02500 | Social Security (shared by taxable) |
| c18300 | amount | 18300 | **Actual SALT (after $10K cap)** |
| e26270 | amount | 26270 | Partnership/S-corp |

### Key Findings — 2026-03-18 Session

**c18300 targeting (major improvement)**:
Replaced uncapped potential SALT targets (e18400/e18500) with actual SALT (c18300)
using SOI A18300 shares — apples-to-apples: Tax-Calculator actual SALT under 2022
law shared using observed 2022 SOI actual SALT by state.

| Metric | e18400/e18500 | c18300 | Change |
|--------|--------------|--------|--------|
| Violations | 85 | 75 | -12% |
| wRMSE avg | 0.479 | 0.289 | -40% |
| wMax avg | 26.3 | 10.6 | -60% |
| %zero avg | 6.2% | 1.3% | -79% |
| Under-used (<0.90) | 27.1% | 3.9% | massive |
| SALT aggregation | -6.23% | -0.18% | 34x better |

**e26270 SOI discrepancy resolved**: The +12.8% was a bug in analysis code
(first AGI bin has -$136B partnership losses; filter incorrectly dropped it).
Actual sum-of-states vs national = -0.3%.

**SALT nz-count infeasibility**: e18400/e18500 nz-count targets caused
PrimalInfeasible in high-SALT states (CA, DC, GA, MD, UT, VA) due to
cross-constraint conflicts with return-count-by-fstatus targets. Now moot
since we target c18300 amounts only.

**Rebase check**: PRs #452-454 only removed unused metadata files.
TMD data unaffected — no regeneration needed.

**Largest violation stat**: Solve summary now shows e.g.
"Largest violation: 5.46%." alongside the violated target count.

### Current Quality Summary (51/51 solved)
```
Total violated targets: 75 (all count targets, no amount violations)
Hit rate: avg=98.8%, min=94.1%
wRMSE: avg=0.289, max=0.805
SALT aggregation: -0.18%
Returns: -0.40%, AGI: -0.74%, Wages: -0.81%, Income tax: -1.02%
```

### Usage
```bash
# Full pipeline:
python -m tmd.areas.prepare_and_solve --scope states --workers 8

# Prepare targets only:
python -m tmd.areas.prepare_and_solve --scope states --stage targets

# Solve only (targets must exist):
python -m tmd.areas.prepare_and_solve --scope states --stage solve --workers 8

# Quality report:
python -m tmd.areas.quality_report
```
