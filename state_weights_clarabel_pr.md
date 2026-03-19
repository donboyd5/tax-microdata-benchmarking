# State Weights Clarabel PR

## Session Notes — 2026-03-18

### Branch: `state-weights-clarabel`

### Commits (15)
1. `0d77d06` — Add Clarabel QP solver and share-based state targeting pipeline
2. `e6be2f7` — Use separate target/weight subfolders for states and CDs
3. `5a1db46` — Exclude PR and US from state target file generation
4. `5c9547c` — Change default SOI area data year from 2021 to 2022
5. `8d8f126` — Add cross-state quality summary report
6. `756ae24` — Print SOI year in prepare and solve stage messages
7. `24d5dc1` — Optimize quality report: load data once, read only needed columns
8. `13eb8cb` — Switch to full recipe, update variable mapping to shared names
9. `3378a4b` — Exclude e26270 and SALT nz-counts from recipe for solver stability
10. `0bdc6d8` — Add largest violation percentage to solve summary output
11. `eb00f68` — Restore e26270 to recipe (SOI discrepancy was analysis bug)
12. `4c06ec6` — Replace e18400/e18500 with c18300 (actual SALT) targeting
13. `308edda` — Document SALT nz-count limitation in recipe comments
14. `004c3f5` — Clarify cross-state aggregation section shows selected variables
15. `f692509` — Use preferred wording for cross-state aggregation header

### PR Strategy (revised 2026-03-18)
- **Two PRs to upstream** (mirroring national reweighting approach):
  - PR 1: Target preparation pipeline (prepare/ modules, recipe, mapping CSV)
  - PR 2: Solver + quality report (builds on PR 1's target files)
- **Two branches**: Branch 1 = target prep only, Branch 2 = solver on top
- User will run the actual push/PR commands

### Current Recipe (12 combos, 119 targets/state)
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

**Double-scaling fix (e00900/e26270)**: Rebased onto upstream
`fix-double-scaling-e00900-e26270` branch and regenerated TMD data.
E00900 and E26270 were 35.5% overstated due to double application of
growth factors. State weighting results essentially unchanged (as expected
— sharing formula is self-correcting), but input data is now correct.

### Current Quality Summary (51/51 solved, after double-scaling fix)
```
Total violated targets: 83 (mostly count targets, 2 small amount violations)
Hit rate: avg=98.6%, min=93.3%
wRMSE: avg=0.294, max=0.827
SALT aggregation: -0.19%
Returns: -0.58%, AGI: -0.74%, Wages: -0.81%, Income tax: -1.01%
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

---

## Session Notes — 2026-03-19

### PR Strategy (revised)
- **Pre-PR #462**: Merged — removed old R/Quarto state target preparation files
- **PR 1 branch**: `prepare-state-targets` — target preparation pipeline
  - Standalone CLI: `python -m tmd.areas.prepare_targets --scope states`
  - Includes base recipe targets + extended targets (SOI-shared, Census-shared, credits)
  - OA share rescaling (51 states sum to 1.0)
  - Tests in `tests/test_prepare_targets.py`
- **PR 2 branch**: `solve-for-state-weights` — solver + quality report (future)
  - Standalone CLI: `python -m tmd.areas.solve_weights --scope states --workers 8`
- Tolerance changed from 0.4% to **0.5%** (matching national reweighting)

### Extended Targets (ported from area-weighting-overhaul)
Added to base recipe targets in a single-pass write:

| Category | Variables | Source | Stubs | Targets/state |
|----------|-----------|--------|-------|---------------|
| SOI-shared | e01700, c02500, e01400, capgains_net, e00600, e00900, c19200, c19700 | SOI amount shares | 5–10 | 48 |
| Census-shared | e18400, e18500 | Census S&L tax collections | 5–10 | 12 |
| Aggregate | eitc, ctc_total | SOI shares (no AGI breakdown) | all | 4 |

Total: 119 base + 64 extended = **183 targets/state**

### OA Share Rescaling
SOI "Other Areas" (~0.5% of returns) excluded. Raw shares (state/US) rescaled
so 51-state shares sum to 1.0. Implemented in `compute_soi_shares()`.

### Key Finding: Dual Variable Analysis (constraint cost)
**Clarabel's dual variables reveal which constraints are expensive to satisfy.**
A target can be perfectly hit but still cause massive weight distortion if the
solver strains to keep it within tolerance. This is invisible from violations alone.

Cross-state dual analysis (8 states: MN, NY, WY, DC, ND, CA, TX, IL):

| Target Type | Avg Dual Cost | Share |
|-------------|--------------|-------|
| c00100 return counts $1M+ (all fstatus) | **~40M each** | **100%** |
| e00200 wage nz-count $1M+ | 3.2 | ~0% |
| All extended targets combined | <0.2 | ~0% |

**The $1M+ AGI bin filing-status counts are essentially the only expensive
constraints.** NY's duals are 323 million. Extended targets (capgains, pensions,
charitable, etc.) are essentially free — near-zero dual cost.

Secondary: $500K–$1M filing-status counts have dual costs ~0.5, noticeable
but 8 orders of magnitude less than $1M+.

**Implication**: Dropping count targets in the $1M+ bin would eliminate virtually
all constraint cost while keeping all amount targets and extended targets.

**NOTE**: Dual variable analysis should be considered for national reweighting too.

### Comparison: 119 vs 183 targets (before recipe refinement)
| Metric | 119 targets | 183 targets |
|--------|------------|------------|
| wRMSE avg | 0.295 | 0.596 |
| %zero avg | 1.4% | 7.3% |
| Under-used (<0.90) | 3.7% | 28.4% |
| Returns aggregation | -0.35% | **-0.13%** |
| AGI aggregation | -0.73% | **-0.13%** |
| Income tax aggregation | -1.01% | **+0.02%** |
| Cap gains aggregation | -3.41% | **-2.51%** |

Aggregation improved substantially; weight distortion increased due to
expensive count targets in $1M+ bin (not the extended targets).

### Recipe Refinement (completed)
Excluded filing-status count targets (fs=1,2,4) and wage nz-count from
$1M+ AGI bin. Kept total return count (fs=0) for $1M+ as anchor.

| Metric | 119 base | 183 all counts | **178 final** |
|--------|---------|---------------|--------------|
| Violations | 81 | 118 | **33** |
| Largest violation | 31.6% | 17.7% | **0.50%** |
| Hit rate avg | 98.6% | 98.7% | **99.6%** |
| wRMSE avg | 0.295 | 0.596 | **0.594** |
| %zero avg | 1.4% | 7.3% | **7.3%** |
| NY status | AlmostSolved | AlmostSolved | **Solved** |
| Returns agg | -0.35% | -0.13% | **-0.14%** |
| AGI agg | -0.73% | -0.13% | **-0.13%** |
| Income tax agg | -1.01% | +0.02% | **+0.01%** |

All 51 states solved, all amount targets met, remaining 33 violations
are total return counts at exactly 0.50% (boundary).

### PR 1 Final State: `prepare-state-targets`
Single commit on master — Add Python state target preparation pipeline
- 36 files changed, ~3,300 insertions (includes SOI data file renames)
- ~178 targets/state (114 base + 64 extended)
- Tests: 10 tests in `tests/test_prepare_targets.py`
- Format/lint clean, all 10 tests pass
- Pushed upstream, Martin reviewing

Directory restructure in this PR:
- SOI state data: `targets/prepare/prepare_states/data/` → `prepare/data/soi_states/`
- Recipes: `targets/prepare/target_recipes/` → `prepare/recipes/`
- Old prepare_states/ infrastructure removed

Cleanup PRs merged before this:
- PR #462: Removed old R/Quarto state preparation files
- PR #463: Removed old CD pipeline, weight examination, stale recipes;
  tracked CD crosswalk files
- PR #464: Hotfix — restored xx test fixtures accidentally deleted in #463

### Next: PR 2 (`solve-for-state-weights`)
Branch off `prepare-state-targets`. Solver + quality report.
- Standalone CLI: `python -m tmd.areas.solve_weights`
- Clarabel QP solver with 0.5% tolerance
- Batch runner with parallel workers
- Quality report (cross-state aggregation, weight diagnostics)
- Key design decisions to carry forward:
  - Dual variable analysis capability (useful for national too)
  - Elastic slack for infeasibility handling
  - AREA_CONSTRAINT_TOL = 0.005 (matching national)

### SOI N2 = "Number of individuals"
Per SOI documentation, N2 counts all people on tax returns (filers +
spouses + dependents). Post-TCJA, derived from filing status and
dependent indicators. N2 = 293.6M (88% of Census 333.3M). Correlates
with Census at r=0.9998. Currently using Census for XTOT pop_share;
revisit if needed.

---

## Session Notes — 2026-03-19 (continued)

### PR 2 Branch: `solve-for-state-weights`
Created off `prepare-state-targets`. Ported solver pipeline from
`state-weights-clarabel` branch.

#### New files added:
| File | Description |
|------|-------------|
| `tmd/areas/create_area_weights_clarabel.py` | Clarabel QP solver (626 lines) |
| `tmd/areas/batch_weights.py` | Parallel batch runner with worker-cached TMD data |
| `tmd/areas/quality_report.py` | Cross-state quality summary report |
| `tmd/areas/solve_weights.py` | Standalone CLI: `python -m tmd.areas.solve_weights` |
| `tests/test_solve_weights.py` | 11 tests (solver on xx, log parser, scope parsing, filtering) |

#### Key changes from `state-weights-clarabel`:
- **Tolerance**: `AREA_CONSTRAINT_TOL = 0.005` (0.5%, matching national)
- **Default paths**: `targets/states/` and `weights/states/` (matching PR 1 structure)
- **CLI split**: `prepare_targets.py` (PR 1) + `solve_weights.py` (PR 2) replace
  the combined `prepare_and_solve.py`
- **No old R/Quarto files**: those were cleaned up in pre-PR commits

#### Usage:
```bash
# Solve all states (8 parallel workers):
python -m tmd.areas.solve_weights --scope states --workers 8

# Specific states:
python -m tmd.areas.solve_weights --scope MN,CA,TX --workers 4

# Quality report:
python -m tmd.areas.quality_report
```

#### Status:
- Format/lint clean
- All 148 tests pass (11 new + 137 existing)
- 4 commits on branch

### Parameter Sweep Results (2026-03-19)

`weight_penalty` has **no effect** — only increases violations.
`multiplier_max` is the **only effective lever** for exhaustion.

| mult_max | Violations | MaxViol% | wRMSE | MaxExh |
|----------|-----------|---------|-------|--------|
| 100 | 33 | 0.50% | 0.594 | 25.2 |
| **25** | **35** | **0.50%** | **0.609** | **16.6** |
| 15 | 210 | 100% | 0.601 | 11.9 |
| 10 | 402 | 100% | 0.625 | 8.8 |

Set `AREA_MULTIPLIER_MAX = 25.0` as default. Sweet spot: 2 extra
violations, 34% lower max exhaustion, single pass.

Two-pass exhaustion limiting tested (`--max-exhaustion 5`) but
**not recommended** — pass 2 caused 8,979 violations (100% max).
Proportional cap scaling too aggressive.

### Commits on `solve-for-state-weights`:
1. `d3021fb` — Core solver pipeline
2. `ef68841` — Target count in batch progress
3. `f73d41f` — Exhaustion limiting + parameter sweep
4. `4d899ad` — Set mult_max=25, add lessons doc
