**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Congressional District Targets and Weights Pipeline
*Branch: `improve-potential-targets-structure`*
*Created: 2026-03-21*

## Background

Previously we improved the weighting process for national targets for the TMD file and overhauled the state area weights process completely. Related session notes: `area_weighting_notes.md` and `nlp_reweighting_session_notes.md`.

Now we are going to develop area targets and weights for congressional districts, for the 2022 tax year.

## Current State Pipeline (on master)

```bash
make clean
make data                                          # makes national TMD data; all non-skipped tests pass
python -m tmd.areas.prepare_targets --scope states  # create targets for 51 states/DC
python -m pytest tests/test_prepare_targets.py -v   # all target tests should pass
python -m tmd.areas.solve_weights --scope states --workers 8  # create the state weights
python -m pytest tests/test_solve_weights.py -v     # all weight tests should pass
python -m tmd.areas.quality_report                  # prints to the log
```

## Congressional District Challenges

Congressional district targets and weighting should be very similar to the state process. However, there are additional challenges that require data exploration before we begin:

- **SOI data structure differences:** CDs may have a different number of income ranges than states, and may have different variables. We need a complete inventory of these differences.
- **Coverage gaps:** We need to determine which congressional districts are present in the IRS data, how many might be missing, and what their population adds up to compared to the relevant national population.
- **District boundary crosswalk:** Both 2021 and 2022 SOI CD data use 117th Congress boundaries. We need a crosswalk to 118th Congress (current) boundaries using `geocorr2022_2607106953.csv`. Code that created a crosswalk from this file should exist somewhere in Git history.
- **Raw SOI CD data:** The raw congressional district SOI data should be somewhere in Git as well.

After the analysis, we may need to make minor adjustments to the proportion-of-national-population approach we used for states.

## Exploration Findings (2026-03-21)

### Data Structure
- Both 2021 and 2022 CD data have **9 AGI bins** (stubs 1–9), confirming `CD_AGI_CUTS` in constants.py.
- 2022 CD: 161 data columns; 2022 state: 161 data columns. Nearly identical — CD has `A00101` (state doesn't), state has `MVITA` (CD doesn't).
- 14 columns changed between 2021→2022 CD data (some dropped, some added — normal year-over-year SOI changes).

### Coverage
- **All 436 CDs are present** in both 2021 and 2022 data. The 8 at-large/single-CD states (AK, DC, DE, MT, ND, SD, VT, WY) use `CONG_DISTRICT=0` instead of `1` — recode to `1` in the pipeline.
- For multi-CD states, CD counts match 117th Congress expectations exactly.
- CDs sum exactly to the CD file's own US aggregate (ratio = 1.0000 for N1, AGI) once at-large states are included.

### CD File vs State SOI File
- The CD and state files are **different SOI products** with slightly different coverage.
- CD file state totals are ~98.3% of state file state totals for N1 (returns), ~98.3% for AGI on average.
- At-large states match almost exactly on returns (ratio ~1.0000) but can differ on AGI (e.g., MT = 0.9502).
- Multi-CD states range from 0.975 to 0.999.
- **Decision: Use CD file's own totals as denominators** for share computation, not state file totals. This keeps shares internally consistent and summing to 1.0.

### Crosswalk
- Geocorr crosswalk has 1,448 rows mapping 117th→118th Congress districts.
- Includes population-weighted allocation factors (`afact2` = cd117-to-cd118).
- Has a label/header row that needs skipping.
- Many CDs split across boundaries (e.g., NC-03 splits with factor 0.858).

### Comparison: States vs CDs vs Counties

| Attribute | States | CDs | Counties |
|-----------|--------|-----|----------|
| Count | 51 | 436 | 3,143 |
| AGI stubs | 10 | 9 | 8 |
| Total row in data | Yes (stub 0) | Yes (stub 0) | Separate file |
| Crosswalk needed | No | Yes (117th→118th) | No |
| Coverage | 100% | 100% (with at-large recode) | 100% |
| Variables | 161 data cols | 161 data cols | 161 data cols |
| Smallest area (returns) | WY ~281K | ~70K | Loving TX: 40 |

### Data Locations
- 2021 CD data: `tmd/areas/targets/prepare/prepare_cds/data/data_raw/congressional2021.zip` (contains `21incd.csv`)
- 2022 CD data: `tmd/areas/targets/prepare/prepare_cds/data/data_raw/22incd.csv`
- 2022 CD docguide: `tmd/areas/targets/prepare/prepare_cds/data/data_raw/22incddocguide.docx`
- Exploration script: `tmd/areas/explore_cd_data.py`

## Performance Analysis and Optimization (2026-03-21)

### Solver Timing
- Per-area solve: ~26 seconds (dominated by Clarabel QP solver, not matrix construction).
- 16 workers is the sweet spot on Ryzen 9 9950X (16C/32T). 28 workers is slower due to memory contention.

### Projected Times (16 workers)

| Scope | Areas | Estimated time |
|-------|-------|---------------|
| States | 51 | ~3.5 min |
| CDs | 436 | ~12 min |
| Counties | 3,143 | ~1.5 hrs |

### Optimization A+B: Constraint Matrix (branch `optimize-constraint-matrix`)
- **A:** Pre-cache DataFrame column arrays — avoid repeated `vardf[col].astype(float).values` in per-target loop.
- **B:** Build sparse B matrix directly using COO format — avoid allocating dense n_records × n_targets intermediate (~0.3 GB).
- Result: ~3-5% wall-time improvement, less memory churn, all tests pass, identical results.
- Benefit grows with more targets per area (CDs and especially counties).
- **To merge:** This branch should be merged to master before starting the CD pipeline branch.

### Optimization C (not yet implemented): Relax Clarabel Tolerances
- Currently `tol_gap_abs = tol_feas = 1e-7`. Relaxing to `1e-5` could cut solver iterations by 20-30%.
- Estimated savings: ~6-8s per area, which matters for counties (~40 min saved).
- Risk: May increase target violations slightly. Needs testing.

### Other Speed Considerations for Counties
- Conservative recipe (fewer targets) → smaller QP → faster solve.
- Tiered recipe by county size (full recipe for 50K+ returns, reduced for <5K).
- Shared memory (Ray) could reduce per-worker memory from 0.55 GB to shared.

## CD Pipeline Implementation (2026-03-22)

### Branch: `cd-pipeline` (off master)

Implemented the full CD target preparation and weight solving pipeline.

### New Files
- `tmd/areas/prepare/soi_cd_data.py` — CD SOI data ingestion, 117th→118th crosswalk, base targets
- `tmd/areas/prepare/recipes/cds.json` — CD recipe (9 AGI bins, same variables as states)
- `tmd/areas/prepare/recipes/cd_variable_mapping.csv` — Variable mapping (identical to state)
- `tmd/areas/prepare/data/soi_cds/22incd.csv` — CD SOI data in canonical location

### Modified Files
- `tmd/areas/prepare/constants.py` — Added `AreaType.CD`, `CD_AGI_CUTS` (9 bins), `SOI_CD_CSV_PATTERNS`, `AT_LARGE_STATES`, helper functions
- `tmd/areas/prepare/target_sharing.py` — Added `compute_cd_soi_shares`, `build_cd_shares_targets`, CD branch in `prepare_area_targets`
- `tmd/areas/prepare/target_file_writer.py` — Added CD areatype dispatch
- `tmd/areas/prepare_targets.py` — Added `prepare_cd_targets()`, `--scope cds` CLI
- `tmd/areas/create_area_weights.py` — Added `CD_TARGET_DIR`, `CD_WEIGHT_DIR`
- `tmd/areas/solve_weights.py` — Added `solve_cd_weights()`, `--scope cds` CLI

### Key Design Decisions
- **XTOT uses N2** from CD SOI file (exemptions, proxy for population)
- **Shares use CD file's own totals** as denominators (internally consistent)
- **117th→118th crosswalk** properly handles MT (1→2 districts) and 7 other at-large states
- **At-large recode:** SOI `CONG_DISTRICT=0` recoded to district 1; variable renamed to `cd117_district` to avoid confusion with documentation meaning
- **AGI and returns sum exactly** to TMD national totals (ratio = 1.000000)
- **102 targets per CD** (vs 179 for states — no extended targets yet)

### Solver Tuning and Infeasibility Investigation (2026-03-22)

Initial run with default 25x multiplier cap: 16 of 436 CDs PrimalInfeasible.

**Root cause analysis:**
- 15 of 16 failures caused by `e02400` (Social Security) target in the negative-AGI bin. These CDs have extremely high SS income per negative-AGI return ($40K-$314K) — the national microdata lacks enough records with this profile to satisfy the constraint within 25x multiplier bounds.
- 1 failure (FL-28) caused by `e26270` (partnership/S-corp) profile incompatible with national microdata in low-AGI bins.
- AZ-01 and TX-07: no individually unreachable targets, but constraint interactions made the system infeasible.

**Fixes applied (uniform rules for all CDs):**
1. **Recipe:** Excluded `e02400` from AGI stub 1 (negative-AGI bin) via `agi_exclude: [1]` in `cds.json`.
2. **Multiplier cap:** Raised from 25x to 50x (`CD_MULTIPLIER_MAX`). States stay at 25x.
3. **Unreachable target detection:** `_drop_impossible_targets` now checks whether each target is achievable within multiplier bounds, not just whether the B matrix row is all zeros. Auto-drops with diagnostic log message.
4. **Variable-bin slack penalties:** e02400, e00300, e26270 amounts in stubs 1-3 and filing-status counts in stubs 1-2 get reduced slack penalty (1e3 vs 1e6). Solver relaxes these before distorting weights.
5. **LP feasibility pre-check:** Fast linear program runs before QP to detect infeasibility early and identify which constraints are problematic.
6. **Solver override framework:** `solver_overrides.py` supports per-area parameter customization via centralized YAML file. Designed for future developer mode.

### Recipe Evolution and Final Results

**Lean recipe (78 targets, 50x cap, Census population XTOT):**

Key recipe decisions:
- **Dollar amounts** (count=0): all 7 income variables × 9 AGI bins (minus e02400 in stub 1). These carry the geographic income story.
- **Total return counts** (count=1, fs=0): bins 4-9 only ($25K+), plus one all-bins total. Low-AGI bins (1-3) dropped — small cells cause infeasibility.
- **Filing-status totals** (count=1, fs=1/2/4): one target each across all bins. Anchors demographic mix. Sum implicitly constrains total returns.
- **Wage nz-count** (count=2): bins 2-8.
- **XTOT**: Census 2020 population from geocorr crosswalk (not SOI N2). Matches state pipeline approach. Fixed -11% returns aggregation error.

Previous recipe attempts:
- 102 targets (per-bin filing-status counts): 16 failures, 730% max violation
- 105 targets (added totals to old recipe): 0 failures but 1,680 violated targets
- 77 targets (lean, no total returns): 44 violated, but -11% returns gap
- **78 targets (lean + total returns + Census pop): 37 violated, 0.50% max, -0.31% returns**

| Metric | Initial (25x, 102 tgts) | Final (50x, 78 tgts) |
|--------|-------------------------|----------------------|
| Failed | 16 | **0** |
| Violated targets | 1,648+ | **37** |
| Largest violation | 730% | **0.50%** |
| Returns aggregation | -14.5% | **-0.31%** |
| AGI aggregation | -6.5% | **+0.25%** |
| Wages aggregation | -5.2% | **+0.26%** |
| SALT aggregation | -5.5% | **+0.44%** |
| Capital gains | -19.7% | **-1.72%** |
| Income tax | -7.2% | **+0.54%** |
| Solve time (16 workers) | ~17 min | **~27 min** |

### Production Architecture (planned)

- **Developer mode (offline, iterative):** Runs LP feasibility checks on all areas, applies relaxation cascade (drop unreachable → reduce slack → drop targets → raise tolerance), writes per-area overrides YAML. Runs once per new data vintage.
- **Production mode (single pass):** Reads override file, solves all areas in one pass. Guaranteed to succeed.
- Override file is committed to repo, not generated at runtime.

### Target Specification Redesign (planned)
- Replace JSON recipe + crossing with a flat CSV target spec file where each row is one target (varname, type, scope, fstatus, agilo, agihi). What you see is what you get — no crossing, no exclude lists, no area-type-specific stub numbering.
- Separate solver params file (CSV or YAML) for per-area overrides.
- Self-documenting, easy to modify, data-driven.

### Future Work
- See `future_state_consistency_pr.md` for potential state pipeline alignment changes
- Target specification redesign (flat CSV, no crossing)
- Extended targets for CDs (SOI-shared, credits) — needs Census SALT data adaptation
- Developer mode auto-relaxation implementation
- Multi-perspective bystander reporting (% of CD total, not just % of bin target)
- Optimization A+B (stashed on `optimize-constraint-matrix` branch)
- Optimization C (relaxed Clarabel tolerances)

## County Analysis

See `county_weighting_notes.md` for detailed county feasibility analysis. County data is stored on a separate `county-data` branch pushed to origin fork, not merged into CD or master branches.
