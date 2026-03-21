# PR: Add Python IRS-to-targets pipeline

**Branch:** `add-python-targets-pipeline`
**Depends on:** PR #449 (delete-r-targets-pipeline, already merged)

## Summary

Adds a pure Python pipeline that produces `soi.csv` from the IRS SOI Excel
files already in the repo. `soi.csv` defines the available national
reweighting targets; `reweight.py` selects which ones to use.

### Purpose and philosophy

The repo ships a vetted `soi.csv` — most users never need to run the pipeline.
The pipeline exists for three reasons:

1. **Auditability** — every number in `soi.csv` traces back to a specific IRS
   Excel cell through documented, readable code.
2. **Reproducibility** — anyone can regenerate `soi.csv` from the IRS Excel
   files already in the repo.
3. **Maintainer workflow** — when a new tax year is added, maintainers run the
   pipeline, verify the output, and commit the updated `soi.csv`. See
   [docs/adding_a_new_year.md](tmd/national_targets/docs/adding_a_new_year.md).

The pipeline is **not** part of `make data`. Regular users run `make data` as
before and get the shipped `soi.csv`. Pipeline tests live in
`tests/national_targets_pipeline/` and are excluded from `make data` via
`--ignore`.

### Pipeline stages

1. **`extract_irs_to_csv.py`** — reads raw IRS `.xls` files (xlrd), writes
   `data/extracted/{year}/{table}.csv`. One-time step; skips files that already
   exist (`--overwrite` to regenerate). Captures full hierarchical column
   headers including merged/spanner cells for auditability.

2. **`build_targets.py`** — assembles extracted CSVs into
   `data/irs_aggregate_values.csv`. Preserves all cross-table data including
   redundancy and minor integer differences (this file is the complete audit
   trail).

3. **`potential_targets_to_soi.py`** (existing, lightly modified) — maps IRS
   variables to TMD names, deduplicates to one row per optimizer key (tab11
   wins when sources conflict), writes `soi.csv` with deterministic sort order.

### Maintainer workflow

```bash
python -m tmd.national_targets.extract_irs_to_csv --overwrite
python -m tmd.national_targets.build_targets
python -m tmd.national_targets.potential_targets_to_soi

python -m pytest tests/national_targets_pipeline -v

make clean && make data
```

### Key design decisions

- **`irs_aggregate_values.csv` is not deduplicated** — intentionally preserves
  cross-table redundancy so minor differences are visible and auditable.
- **Deduplication happens in stage 3** — `Value` excluded from dedup key so
  ±1 integer differences between tables don't create duplicate rows. Tab11 is
  authoritative.
- **Updated `soi.csv` corrects 8 off-by-one errors** from the old R pipeline.
- **IRS Excel files remain in the repo** as ground truth; extracted CSVs are
  gitignored (regenerate with stage 1).
- **One IRS variable deliberately skipped** — the base itemized deduction
  filer count (`id/count/nz`) has no direct TMD mapping in
  `irs_to_puf_map.json`. This is correct: the same data is already captured
  via the `id_*` sub-component mappings, which produce the
  `itemized_deductions` rows in `soi.csv` (values match exactly).
- **Deterministic sort order** on `soi.csv` output for reproducible diffs.
- **No new global lint disables** — `table_layouts.py` (1,069-line config)
  uses a per-file `pylint: disable=too-many-lines`.

### Tests

79 tests (42 functions, some parametrized) in
`tests/national_targets_pipeline/test_national_targets.py` covering config
sanity, extracted CSV shape and known IRS totals, `irs_aggregate_values.csv`,
and `soi.csv`. These tests are **not** part of `make data`.

## Test plan

- [x] Full pipeline: `extract_irs_to_csv.py --overwrite` -> `build_targets.py`
  -> `potential_targets_to_soi.py`
- [x] `python -m pytest tests/national_targets_pipeline -v` (79 passed)
- [x] `make format && make lint` (clean)
- [x] `make clean && make data` with shipped `soi.csv` (48 passed, 6 skipped)
- [x] `make clean && make data` with pipeline-generated `soi.csv`
  (48 passed, 6 skipped)
