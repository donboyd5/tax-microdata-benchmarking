# State Weights Clarabel PR

## Session Notes — 2026-03-16

### Branch: `state-weights-clarabel`

### Commits (7)
1. `bb56d06` — Add Clarabel QP solver and share-based state targeting pipeline
2. `2333b5a` — Use separate target/weight subfolders for states and CDs
3. `2c87ea6` — Exclude PR and US from state target file generation
4. `c2b7afe` — Change default SOI area data year from 2021 to 2022
5. `7b65f57` — Add cross-state quality summary report
6. `bfd320a` — Print SOI year in prepare and solve stage messages
7. `a9da9a6` — Optimize quality report: load data once, read only needed columns

### Files Changed (14 files, +3,842 lines)
- `tmd/areas/create_area_weights_clarabel.py` — Clarabel QP solver (626 lines)
- `tmd/areas/batch_weights.py` — Parallel batch runner (334 lines)
- `tmd/areas/prepare_and_solve.py` — End-to-end pipeline orchestrator (282 lines)
- `tmd/areas/quality_report.py` — Cross-state quality summary (606 lines)
- `tmd/areas/create_area_weights.py` — USE_CLARABEL bridge (+11 lines)
- `tmd/areas/prepare/target_sharing.py` — Share-based targeting (747 lines)
- `tmd/areas/prepare/target_file_writer.py` — Target CSV writer (304 lines)
- `tmd/areas/prepare/soi_state_data.py` — SOI state data ingestion (315 lines)
- `tmd/areas/prepare/constants.py` — Constants and enums (241 lines)
- `tmd/areas/prepare/census_population.py` — Census population data (175 lines)
- `tmd/areas/prepare/data/state_populations.json` — State population JSON
- `tmd/areas/prepare/__init__.py` — Package init
- `tmd/areas/targets/prepare/target_recipes/states_safe.json` — 91-target recipe
- `tmd/areas/targets/prepare/target_recipes/state_variable_mapping_safe.csv` — Variable mapping

### Key Design Decisions
- Targets and weights use separate subfolders: `targets/states/`, `weights/states/` (ready for `targets/cds/`, `weights/cds/`)
- PR and US are excluded from target generation (50 states + DC = 51 areas)
- Default SOI year is 2022 (matching TAXYEAR)
- Target CSVs are NOT committed — they'll be added with the 161-target recipe in PR 2
- 91 "safe" targets (well-aligned variables with <2% SOI-TMD difference)

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

## PR Message (for upstream PSLmodels PR)

### Title
Add Clarabel QP solver for state area weights

### Body

## Summary
- Replace L-BFGS-B solver with Clarabel quadratic programming solver for state area weight optimization (~6x faster)
- Add Python pipeline for SOI data ingestion and share-based state targeting (area_target = TMD_national x SOI_share)
- Add parallel batch runner with progress reporting
- Add cross-state quality summary report with target accuracy, weight distortion, and national aggregation diagnostics

## What this does
This PR adds a complete state area weighting pipeline:

1. **SOI data ingestion** (`prepare/soi_state_data.py`): Reads SOI state-level CSV data and computes area shares
2. **Share-based targeting** (`prepare/target_sharing.py`): Computes state targets as national TMD totals x SOI shares
3. **Target file writer** (`prepare/target_file_writer.py`): Generates per-state target CSV files from a JSON recipe
4. **Clarabel QP solver** (`create_area_weights_clarabel.py`): Formulates weight optimization as a quadratic program with slack variables for infeasible targets
5. **Batch runner** (`batch_weights.py`): Parallel execution across all 51 areas (50 states + DC) using ProcessPoolExecutor with per-worker TMD data caching
6. **Quality report** (`quality_report.py`): Parses solver logs to produce summary statistics on target accuracy, weight distortion, violations, and cross-state aggregation vs national totals
7. **Pipeline orchestrator** (`prepare_and_solve.py`): CLI entry point with `--stage targets`, `--stage solve`, or `--stage all`

The initial recipe includes 91 "safe" targets — variables with <2% SOI-TMD national discrepancy (AGI, wages, interest, partnership/S-corp income, by AGI stub and filing status).

## Architecture
- Targets and weights use separate subfolders (`targets/states/`, `weights/states/`) to support future CD weighting
- Target CSVs are generated at runtime and not committed to the repo (a follow-up PR will add the extended 161-target recipe and commit final target files)
- The old L-BFGS-B solver is preserved behind a `USE_CLARABEL = True` bridge in `create_area_weights.py`

## Test plan
- [ ] Run `python -m tmd.areas.prepare_and_solve --scope states --workers 8` — all 51 states should solve
- [ ] Run `python -m tmd.areas.quality_report` — verify target accuracy and weight distortion are reasonable
- [ ] Run `make format && make lint` — should pass cleanly
- [ ] Verify existing tests still pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
