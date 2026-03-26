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

### Target Specification Redesign (implemented 2026-03-23)

Replaced JSON recipe + AGI crossing with three clean artifacts:

**Architecture:**
```
SOI data + crosswalks → shares file (stable, rarely recomputed)
                              ↓
TMD data → national sums (recomputed when TMD changes)
                              ↓
shares × national sums → universe of potential targets
                              ↓
spec CSV → select from universe → per-area _targets.csv
```

**Three artifacts, three change frequencies:**
| Artifact | Location | Rows | Changes when |
|----------|----------|------|-------------|
| `cd_target_spec.csv` | `recipes/` | 78 | Recipe tuning |
| `cds_shares.csv` | `prepare/data/` | 57,116 | New SOI vintage |
| `{area}_targets.csv` | `targets/cds/` | 78 × 436 | TMD rebuild or recipe change |

**Key design decisions:**
- **Shares (stable):** SOI geographic distribution saved as `cds_shares.csv`. Depends only on SOI data and crosswalks. `python -m tmd.areas.prepare_shares --scope cds`.
- **Spec (recipe):** Flat CSV, one row per target — WYSIWYG. Includes `description` column for documentation. No crossing, no exclude lists.
- **Targets (volatile):** `target = TMD_national_sum × soi_share`. XTOT stored as fixed target (Census population). Total rows computed as sum of per-bin targets (not total_share × total_sum) for mathematical consistency.
- **Verified:** 436/436 CDs produce identical targets vs old pipeline (worst relative diff: 5×10⁻¹⁰).

**New files:**
- `tmd/areas/prepare_shares.py` — pre-compute SOI shares
- `tmd/areas/prepare/recipes/cd_target_spec.csv` — flat CD spec (78 targets)
- `tmd/areas/prepare/recipes/state_target_spec.csv` — flat state spec (179 targets)
- `tmd/areas/prepare/data/cds_shares.csv` — pre-computed CD shares

**Variable name mapping (SOI → TMD):**
- SOI raw `A00100` → SOI base `00100` → TMD `c00100`
- Mapping defined in `ALL_SHARING_MAPPINGS` in `constants.py`
- User-facing spec uses TMD names only; SOI mapping is internal

### Quality Report Improvements (2026-03-23)

Enhanced `quality_report.py` with:
- **Auto-save to file:** `--output` flag, defaults to `quality_report.txt` in weight dir
- **Scope/timestamp header:** `[cds]` label, generation time, cumulative + wall-clock solve time
- **Aggregate multiplier distribution:** Combined histogram across all area-record pairs, shows old (x=1) vs optimized distribution with cumulative percentages
- **Weight distribution by AGI stub:** National vs sum-of-areas returns and AGI per bin, with change %
- **Per-bin bystander analysis:** Checks all variable × AGI bin combos, marks T (targeted) vs . (dropped/untargeted), sorted by distortion. CD results: 73 targeted bins avg 0.13% distortion; 69 dropped bins avg 5.0%
- **Top-N per-area detail:** Shows all areas for states (≤60), top 20 by violations/wRMSE for CDs/counties, with omitted count

### Extended Targets for CDs (2026-03-23)

Added total-only (all-bins aggregate) extended targets, bringing spec from 78 → 92 targets.

**New extended targets (14 rows, all total-only):**
- 8 SOI-shared amounts: e01700 (taxable pensions), c02500 (taxable SS), e01400 (IRA), capgains_net, e00600 (dividends), e00900 (business income), c19200 (mortgage ded), c19700 (charitable)
- 2 SALT components: e18400 (income/sales), e18500 (real estate) — using SOI CD data as proxy for Census (Census not available at CD level)
- 2 credits amount + 2 credits nz-count: eitc, ctc_total

**SALT approach for CDs:**
- States target both c18300 (restricted/capped SALT via SOI shares) and e18400/e18500 (available/uncapped SALT via Census shares)
- CDs can only do SOI-based: c18300 in base recipe + e18400/e18500 using SOI CD columns (a18425, a18500) as proxy
- This is good enough for geographic distribution within the SOI file's coverage

**Implementation:**
- Extended `EXTENDED_SHARING_MAPPINGS` in `prepare_shares.py` — defines the 14 new variable mappings
- One-to-many SOI→TMD mapping: e01500 and e01700 both use SOI 01700; e02400 and c02500 both use SOI 02500
- Added `capgains_net` synthetic variable to `compute_tmd_national_sums` (p22250 + p23250)
- `ctc_total` and `eitc` already exist in cached_allvars.csv
- CTC shares use SOI 07225 (nonrefundable CTC) as the geographic distribution basis; derived `_add_ctc_total` sums 07225 + 11070 in base_targets

**Test-solve results (4 CDs):**
- AL01, AK01: 92/92 targets hit
- CA52: 91/92 (1 minor violation at 0.50%)
- NY12: 86/92 (6 violations, RMSE 9.3) — already the hardest CD, extra targets stress it further

**Incremental extension plan:**
1. ✅ Total-only extended targets (done, 92 targets)
2. ✅ Developer mode toolkit (difficulty table, dual analysis, auto-cascade)
3. ✅ Per-bin cap gains ($100K+): 3 targets, +1.5s, all hit
4. ✗ Per-bin credits (EITC/CTC bins): rejected — 60-90% gap, 7x solve time explosion
5. Final spec: 95 targets (78 base + 14 ext totals + 3 capgains bins)

### Developer Mode Implementation (2026-03-23)

Built as a toolkit of diagnostics rather than a fully automated system:

**Tools:**
- `--difficulty AREA`: Target difficulty table — gap from proportionate share per target. Most useful single diagnostic.
- `--dual AREA`: Shadow price analysis — solves area and shows which constraints are most expensive to satisfy.
- `--lp-only`: LP feasibility check across all areas (fast).
- Full auto-cascade: iterative relaxation for batch override generation.

**Key findings from difficulty/dual analysis:**
- Per-bin credit targets have 60-90% gap from proportionate AND pull weights in opposite directions (EITC: +86%, CTC: -89% in same bin) → solver explodes
- Cap gains per-bin targets feasible (~50% gap, upper stubs only, concentrated records)
- NY-12 (Manhattan) mean gap = 255%, max = 3,384% — an extreme outlier
- Solve time scales super-linearly with target count AND target difficulty, not just count
- Dense constraint rows (all-bin targets) are more expensive but still worthwhile for aggregate control

**Developer workflow documented in `AREA_WEIGHTING_GUIDE.md`:**
1. Identify high-value targets by policy importance
2. Run difficulty tables on representative easy/hard areas
3. Test incrementally on single areas
4. Run dual analysis on problem areas
5. Full batch + quality report
6. Iterate

See `cd_target_difficulty_analysis.md` for detailed comparison of AL-01, NY-12, TX-20, MN-03.

### Solve Time Evolution

| Spec | Targets | All-bin rows | Avg/area (AL01) | Wall@16 | Notes |
|------|---------|-------------|-----------------|---------|-------|
| Base | 78 | 5 | 7s | ~18 min | Original lean recipe |
| +ext totals | 92 | 19 | 12s | ~34 min | +14 total-only extended |
| +cg bins | 95 | 19 | 14s | ~54 min | +3 capgains upper bins |
| +EITC bins | 101 | 19 | 14s | ~54 min est | +6 EITC per-bin (cheap) |
| +credit bins | 107 | 19 | 92s | ~100 min est | Rejected — CTC bins too expensive |

### Developer Mode Results (107-target spec, 2026-03-23)

Full auto-relaxation cascade on all 436 CDs:
- **Level 0 (default):** 337 areas (77%) — solved with no changes
- **Level 3 (drop targets):** 76 areas (17%) — needed 1-8 targets dropped
- **Level 4 (raise cap):** 4 areas (1%) — needed higher multiplier cap
- **Level 5 (raise tolerance):** 19 areas (4%) — needed tolerance relaxation
- **All 436 areas: 0 violations** after overrides applied
- Developer mode time: ~224 min (13,455s)
- Override YAML: `tmd/areas/prepare/recipes/cd_solver_overrides.yaml`

### 95-Target Batch Solve Results (2026-03-23)

| Metric | 78 targets | 92 targets | 95 targets |
|--------|-----------|-----------|-----------|
| Failed | 0 | 0 | 0 |
| Areas with violations | 37 | 47 | 49 |
| Total violated targets | 37 | 52 | 88 |
| Largest violation | 0.50% | 0.50% | 0.50% |
| Wall time (16 workers) | ~27 min | ~52 min | ~54 min |

### EITC vs CTC Per-Bin Analysis

**Initial finding (2026-03-23):** Adding 6 CTC per-bin targets caused
solver explosion (14s → 88s for AL01). Diagnosed as "fundamental
conflict" between EITC and CTC eligibility profiles in the same bins.

**Corrected finding (2026-03-24):** The root cause was a **duplicate
shares bug** in `_add_ctc_total()` in `prepare_shares.py`. The function
added combined CTC (07225 + 11070) rows but didn't remove the original
07225 rows, producing two conflicting shares for every CTC target. The
solver saw contradictory constraints for the same variable.

After fixing the bug and regenerating shares:
- 107 targets (including 6 CTC per-bin) solve all 436 CDs
- 69 total violated targets (max 0.50%)
- ~55 min wall time (16 workers)
- AL01 solves in ~34s (not 88s)

**EITC/CTC co-occurrence is real but not a solver problem.** At AGI
$25K-$50K, 97.5% of EITC recipients also have CTC. This reflects tax
law: qualifying children under 17 trigger both credits. But the
geographic shares genuinely differ because CTC extends to much higher
incomes (phase-out at $200K/$400K) while EITC phases out at $43K-$59K.
With correct (non-duplicate) shares, the solver handles this fine.

**Decision:** Include both EITC and CTC per-bin targets. Final spec:
107 targets.

### 107-Target Batch Solve Results (2026-03-24)

| Metric | 78 targets | 95 targets | 107 targets |
|--------|-----------|-----------|-------------|
| Failed | 0 | 0 | 0 |
| Areas with violations | 37 | 49 | 46 |
| Total violated targets | 37 | 88 | 69 |
| Largest violation | 0.50% | 0.50% | 0.50% |
| Wall time (16 workers) | ~27 min | ~54 min | ~55 min |

Note: 107-target results are BETTER than 95 on violated targets (69 vs
88) because the CTC shares fix also improved other CTC-related targets.

### Optimization Benchmarks (2026-03-24)

Tested on 49 CDs (every 9th area), 16 workers, 107 targets:

| Optimization | Wall time | Speedup | Quality impact |
|-------------|-----------|---------|---------------|
| Baseline | 397.7s | — | 7 violations |
| A+B (matrix construction) | 388.2s | +2.4% | Identical |
| C (tol 1e-7 → 1e-5) | 388.4s | +2.3% | 92 violations (13x worse) |
| OSQP (alt solver) | 325.9s | — | 40 violations, 50K iter (max) |

**Decision:** None applied. A+B adds complexity for negligible gain.
C degrades quality. OSQP can't converge. Clarabel at 1e-7 tolerance
is the right choice. ~55 min for 436 CDs is acceptable.

See `archive/optimization_ab_benchmark.md` for details.

### SOI Data Bug: A59664 Unit Error (2026-03-24)

Column A59664 (EITC amount, 3+ qualifying children) in the 2022 CD
SOI file is in dollars, not $1,000s like all other amount columns.
Workaround applied in `soi_cd_data.py` (divide by 1000 on ingestion).
Email draft for SOI: `soi_a59664_unit_error_email.md`. State file is
not affected.

### PR Strategy (2026-03-24)

Four PRs, sequenced by dependency:

1. **Solver robustness** — range-based feasibility, per-constraint
   penalties, LP pre-check (state-affecting)
2. **Spec-based targets** — shares + spec architecture, constants,
   target pipeline (additive)
3. **Quality report** — multiplier histograms, bystander analysis,
   output options (diagnostic only)
4. **CD pipeline** — SOI data, crosswalk, solver, developer mode,
   documentation (CD-only)

### PR Implementation Progress (2026-03-25, updated 2026-03-26)

PRs built as stacked git worktrees in `~/Documents/mixed_projects/`:
- `pr1-solver-robustness/` — PR #470, **merged** 2026-03-26
- `pr2-spec-targets/` — PR #471, **merged** 2026-03-26
- `pr3-quality-report/` — needs rebase onto merged master, then push
- `pr4-cd-pipeline/` — needs rebase onto PR 3, then push

Cleanup: pr1 and pr2 worktrees deleted. Local master updated to
upstream. Branches area-weighting-overhaul deleted (superseded by
cd-pipeline). county-data kept for future reference.

Martin's feedback on PRs 1-2:
- Per-constraint penalties caused extra violations for states (fixed:
  apply only for CDs)
- CD tests should not be in PR 2 (they skip without pipeline data);
  moved to PR 4
- Suggested area-specific Makefile in tmd/areas/ for pipeline commands

Key decisions during PR preparation:

**Per-constraint slack penalties (PR 1):** Initially applied to all areas,
causing 126 state violations vs 35 baseline. Investigation showed weights
were identical to 2e-8 — purely a boundary effect at the 0.50% tolerance.
Fix: apply penalties only for CDs (multiplier_max > 25). States now match
baseline exactly.

**State SALT targeting (PR 2):** Replaced 12 per-bin e18400/e18500 targets
with 2 total-only targets using Census shares. SOI per-bin SALT shares are
distorted by $10K TCJA cap — high earners show $10K deducted, not $50K owed.
Census total-only is more defensible. State targets: 179 → 169.

**Unified CLI routing (PR 2):** Both states and CDs route through
`prepare_targets_from_spec()`. States use Census shares for available SALT
(e18400, e18500) applied at target-generation time, not in the shares file.

**Memory optimization (PR 1):** Sparse COO matrix construction, sparse LP,
column trimming, parent-process TMD preloading. Peak per-worker: 1,244 MB →
798 MB. Prevents OOM with 16 workers on WSL2.

**Fingerprint test:** On-demand reproducibility test comparing integer weight
sums per area. Verified 8-worker and 16-worker results identical.

### Next Steps (2026-03-26)
- Rebase PR 3 onto merged master, push upstream
- Rebase PR 4 onto PR 3, add areas Makefile, push upstream
- Add areas/Makefile with `make states` and `make cds` targets
- PR 5 candidates: XTOT sharing fix, combined parquet files

### Future Work
- See `future_state_consistency_pr.md` for potential state pipeline alignment changes
- PR 5: Share national XTOT by Census proportions (instead of raw Census pop)
- Combined weight/target files (single parquet per scope)
- Per-area weight distribution diagnostic
- Legacy cleanup PR (remove old recipe system after spec pipeline proven)

## County Analysis

See `county_weighting_notes.md` for detailed county feasibility analysis. County data is stored on a separate `county-data` branch pushed to origin fork, not merged into CD or master branches.
