**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Area weighting

---


## Goals

- Extend area data used to create area weights for states and for Congressional districts to 2022; we already have 2021
- Make the process of developing area targets and mapping them to TMD variables smoother and cleaner
- Ensure that either 2021 or 2022 data can be used to estimate each area's share of national data for specific income and other items: 2021 data shares can be used for any national data year, 2022 area data shares can be used for any national data year, and so on. In practice, we'll almost always pair 2021 area data with 2021 national data and 2022 area data with 2022 national data. But the flexibility to use different years is important.
- Make sure we can reproduce files created under the old process with files under the new process
- Make the area weighting optimization considerably faster. Consider converting the objective function to minimizing change from average national weights, subject to constraints, much as we did with national reweighting. This could involve using the Clarabel solver.
- Look for ways to create area weights en masse efficiently -- all 50 states, or all 435 Congressional districts - perhaps benefiting from solver features and also paralellism
- Convert the data preparation from R to python

## Key Design Principles

### A. All targets as shares (eventual goal)
Area targets = national TMD total × (area SOI / national SOI).
SOI provides geographic distribution, TMD provides levels.
Shares computed at variable × AGI-bin × filing-status granularity.
Transition: reproduce R pipeline first, then convert all to shares.

### B. Optimization variable: deviation from population-proportional
x_i = area_weight / (pop_share × national_weight).
x_i = 1 means population-proportional share.
QP minimizes sum((x_i - 1)²) + M·sum(s_j²) subject to target bounds with elastic slack s_j >= 0 (penalty M=1e6) and multiplier bounds [x_min, x_max] (default [0.0, 100.0]).

## Progress

### Completed

**Phase 1: Module structure and constants** (2026-03-12)
- Created `tmd/areas/prepare/` package with `__init__.py` and `constants.py`
- Constants: AreaType enum, AGI cuts (10 state bins, 9 CD bins), SOI file patterns (2015-2022), SHARING_MAPPINGS, STATE_INFO, ALLCOUNT_VARS

**Phase 2: Clarabel area weight optimizer** (2026-03-12)
- Created `tmd/areas/create_area_weights_clarabel.py` (~620 lines)
- Constrained QP mirroring national reweighting: minimize sum((x_i-1)²) with elastic slack
- Default params: tolerance=0.004, multiplier bounds [0.0, 100.0], slack penalty=1e6
- Bridge flag `USE_CLARABEL = True` in `create_area_weights.py`
- Performance on "xx" area: Clarabel ~9s (all 16 targets hit), L-BFGS-B ~7.5s (1 target miss)
- All 50 existing tests pass

**Phase 3: State SOI data ingestion** (2026-03-12)
- Created `tmd/areas/prepare/soi_state_data.py` — reads raw SOI CSVs, pivots to long format, classifies variables, creates derived vars (18400 = 18425 + 18450), scales amounts x1000, adds AGI labels, produces base_targets
- Created `tmd/areas/prepare/census_population.py` — embedded 2021 Census PEP state populations + ACS 1-year CD populations (436 CDs + US)
- Tested: 8 years (2015-2022) loaded, ~98K rows per year, all 54 areas (incl OA), verified NY values

**Phase 4: Target file writer** (2026-03-12)
- Created `tmd/areas/prepare/target_file_writer.py` — JSON recipe parser (strips // comments), variable mapping loader, match frame builder, per-area CSV writer
- Produces output matching format expected by `create_area_weights.py`: varname,count,scope,agilo,agihi,fstatus,target
- Tested: generates 144-147 targets per state across all 53 areas

**Phase 6a: Target sharing pipeline** (2026-03-12)
- Created `tmd/areas/prepare/target_sharing.py` — computes TMD national sums by AGI bin, SOI geographic shares, derives area targets = TMD_sum x SOI_share
- 4 shared variables: e01500<-01700, e02400<-02500, e18400<-18400, e18500<-18500
- `build_enhanced_targets()` combines base + shared targets with sort ordering
- Full pipeline tested end-to-end: SOI -> base_targets -> enhanced_targets -> target files for all 53 areas

**Phase 5: CD SOI data ingestion** (2026-03-12)
- Created `tmd/areas/prepare/soi_cd_data.py` — reads SOI CD CSV from ZIP, classifies record types (US, DC, cdstate, state, cd), pivots to long, creates derived vars, adds AGI labels, builds base_targets with area codes
- Created `tmd/areas/prepare/cd_crosswalk.py` — 117th-to-118th Congress boundary crosswalk module using geocorr data (ready for crosswalk file when provided)
- Updated `census_population.py` with embedded 2021 ACS 1-year CD populations (436 CDs + US total)
- Key finding: BOTH 2021 and 2022 SOI CD data use 117th Congress boundaries. MT=1 CD (not 2), OR=5 CDs (not 6), TX=36 (not 38) in both years. Crosswalk needed for both years to get 118th Congress targets.
- Tested: 2021 data produces 437 areas (436 CDs + US), 721K base_target rows. 2022 data also reads successfully.

**Phase 6b: All-shares targets** (2026-03-12)
- Added `ALL_SHARING_MAPPINGS` to constants.py — 15 variable combos (8 amounts, 3 nonzero counts, 4 allcounts by filing status)
- New `compute_tmd_national_sums_all()` handles count types 0 (amounts), 1 (allcounts via MARS), 2 (nonzero counts)
- New `build_all_shares_targets()` replaces ALL direct SOI values with TMD x SOI_share
- Created allshares variable mapping CSVs for both states and CDs
- Legacy 4-var sharing preserved via `build_enhanced_targets()` for backward compatibility
- Tested: MN produces 166 all-shares targets (same as legacy target count when processed through the target file writer)
- All-shares and legacy target different VALUES but the same target variables/concepts; mean difference ~8% (SOI vs TMD national differences)

**Phase 7: 2022 data extension** (2026-03-12)
- Moved population data from embedded Python dicts to JSON files in `tmd/areas/prepare/data/`
- Added 2022 state populations (PEP vintage 2023, from NST-EST2023-ALLDATA.csv)
- Added 2022 CD populations (ACS 2022 1-year, 118th Congress boundaries, 436 CDs)
- Key: 2022 ACS CD data is on 118th Congress boundaries, but SOI CD data (both 2021 and 2022) is on 117th — so 2021 CD populations used for 117th Congress processing regardless of SOI year
- Tested: full 2022 state pipeline (SOI → base_targets → all-shares) produces 166 targets per state
- Tested: full 2022 CD pipeline produces 151 targets per CD (NY01 example)
- All 50 existing tests pass

**Phase 8: Flexible year pairing** (2026-03-12)
- Added `prepare_area_targets()` orchestrator in `target_sharing.py`
- Supports independent `area_data_year`, `national_data_year`, `pop_year`
- Tested: 2022 SOI shares + 2021 TMD nationals → ~2% geographic shift for MN AGI (expected)
- All 50 existing tests pass

### Performance (measured, Clarabel solver with all-shares targets)
- ~212,000 TMD records per area
- **Per state: 25-35 seconds** (107 targets), ~30s average
- **DC: 203 seconds** (InsufficientProgress, 13 violated targets — special case)
- **50 states + DC + xx = 52 areas with 8 workers: 3.7 minutes total**
- **Estimated 435 CDs with 8 workers: ~35 minutes**
- **Estimated all 485 areas with 8 workers: ~40 minutes**
- 13 of 52 states had some violated targets (mostly small states: AK, ND, SD, VT, WV, WY)

**Phase 9: Batch processing** (2026-03-12)
- Created `tmd/areas/batch_weights.py` with ProcessPoolExecutor
- TMD data loaded once per worker process (not once per area) via `_init_worker()`
- Progress reporting with ETA, violated target tracking
- Area filtering: `--areas states`, `--areas cds`, `--areas all`, or comma-separated
- `--force` flag to recompute all areas even if up-to-date
- Usage: `python -m tmd.areas.batch_weights --workers 8 --areas states --force`
- All 50 existing tests pass

**Phase 10: Validation** (2026-03-12)
- Fixed bug in `target_file_writer.py`: allcount variable filter didn't recognize shared basesoivnames (e.g., `tmd00100_shared_by_soin1` vs `n1`). This caused all count=1 targets (40 per state) to be dropped with allshares mapping. Fixed by extracting SOI part from shared names.
- **Target count validation**: Old and new pipelines produce IDENTICAL target counts:
  - MN state: 147 targets (both old and new)
  - MN01 CD: 128 targets (both old and new)
- **Target value comparison (MN state)**:
  - Previously shared variables (e01500, e02400, e18400, e18500): typically <1% difference, max ~10% for lowest AGI bins of SALT variables
  - Newly shared variables (c00100, e00200, e00300): mostly 1-5% difference (TMD national differs from SOI)
  - e26270 (partnership/S-corp): up to 445% in small AGI bins — expected, volatile variable
  - c00100 counts (lowest AGI bin): 20-40% — TMD return counts differ from SOI by bin
- **Target value comparison (MN01 CD)**:
  - Larger differences overall (3-28%) because old CD targets went through 117th→118th Congress crosswalk
  - XTOT population: 3.24% difference (different Census source)
  - e18400/e18500: 10-28% in some bins
- **Solver comparison**: Both MN and MN01 solved successfully with Clarabel:
  - MN: 146/147 targets hit, 12.4s, multiplier median=0.994, RMSE=0.317
  - MN01: 124/128 targets hit, 14.1s, multiplier median=0.930, RMSE=0.463
- **Weight comparison**: Can't match record-by-record (old TMD had 225K records vs new 212K). Aggregate comparison: weight sums differ by 3-5% (different TMD base), similar distributions. Clarabel produces more zero weights (1.5% state, 7.5% CD) vs old solver (0.1%).

### Potential Next Steps

- **A. End-to-end orchestration**: Single command from raw SOI data → target files → weight files. Currently requires manual Python calls to connect `prepare_area_targets()` → `write_area_target_files()` → `batch_weights.py`.
- **B. CD crosswalk**: CD targets are on 117th Congress boundaries; need geocorr crosswalk data to produce 118th Congress targets via `cd_crosswalk.py`.
- **C. Full batch validation**: Generate targets for all states/CDs from the Python pipeline and solve them all; check for problem areas.
- **D. Integration testing**: Run Tax-Calculator with Clarabel weights for real states/CDs (not just "xx").
- **E. DC as state**: Treat DC as a state rather than CD (user suggestion).
- **F. Upstream prep**: Clean up for eventual PR — remove R dependency, ensure raw data in-repo, documentation.

## Branch

All work is on the `area-weighting-overhaul` branch. Push only to `origin` (donboyd5 fork).

## Files Created/Modified

New files:
- `tmd/areas/prepare/__init__.py`
- `tmd/areas/prepare/constants.py`
- `tmd/areas/prepare/census_population.py`
- `tmd/areas/prepare/soi_state_data.py`
- `tmd/areas/prepare/target_file_writer.py`
- `tmd/areas/prepare/target_sharing.py`
- `tmd/areas/prepare/soi_cd_data.py`
- `tmd/areas/prepare/cd_crosswalk.py`
- `tmd/areas/prepare/data/state_populations.json`
- `tmd/areas/prepare/data/cd_populations.json`
- `tmd/areas/create_area_weights_clarabel.py`
- `tmd/areas/targets/prepare/target_recipes/state_variable_mapping_allshares.csv`
- `tmd/areas/targets/prepare/target_recipes/cd_variable_mapping_allshares.csv`
- `tmd/areas/batch_weights.py`
- `tmd/areas/targets/prepare/validation/` (old R-pipeline MN/MN01 reference files)

Modified files:
- `tmd/areas/create_area_weights.py` (added USE_CLARABEL bridge)
- `tmd/areas/prepare/constants.py` (added ALL_SHARING_MAPPINGS)
- `tmd/areas/prepare/target_sharing.py` (added all-shares pipeline + orchestrator)
- `tmd/areas/prepare/census_population.py` (moved data to JSON, added 2022)
- `tmd/areas/prepare/target_file_writer.py` (fixed allcount filter for shared names)

## Open Items

- Both 2021 and 2022 CD SOI data are on 117th Congress boundaries — need geocorr crosswalk for either year to produce 118th Congress targets
- Decide on Clarabel multiplier bounds for area weights (currently [0.0, 100.0])
- When creating upstream PR: include only necessary source data (not spreadsheets, etc.)

## Resume Instructions

To continue this work in a new session, paste the following:

> Continue the area weighting system overhaul on the `area-weighting-overhaul` branch. Read the session notes at `session_notes/area_weighting_notes.md`. All 10 original plan phases are complete: module structure, Clarabel solver, state/CD SOI ingestion, target file writer, sharing pipelines (legacy 4-var and all-shares), 2022 data, flexible year pairing, batch processing, and validation. Key files: `tmd/areas/prepare/` (Python data pipeline), `tmd/areas/create_area_weights_clarabel.py` (QP solver), `tmd/areas/batch_weights.py` (parallel runner). Potential next steps: end-to-end orchestration, CD crosswalk, full batch validation, integration testing, DC-as-state, upstream prep. Push only to `origin`, never upstream.
