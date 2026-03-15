# PR #1 (donboyd5 fork): Replace R/Quarto IRS pipeline with pure Python

**Branch:** `python-irs-pipeline`
**PR URL:** https://github.com/donboyd5/tax-microdata-benchmarking/pull/1

## Summary

Replaces the previous R/Quarto pipeline for building national reweighting targets with a clean, fully-documented Python pipeline. The pipeline reads IRS SOI Excel files and produces `soi.csv` through three explicit stages.

### Pipeline stages

1. **`extract_irs_to_csv.py`** — reads raw IRS `.xls` files (xlrd), writes `data/extracted/{year}/{table}.csv`. One-time step; skips files that already exist. Captures full hierarchical column headers including merged/spanner cells for auditability.

2. **`build_targets.py`** — assembles extracted CSVs into `data/irs_aggregate_values.csv`. Preserves all cross-table data including redundancy and minor integer differences (this file is the complete audit trail).

3. **`potential_targets_to_soi.py`** (existing, lightly modified) — maps IRS variables to TMD names, deduplicates to one row per optimizer key (tab11 wins when sources conflict), writes `soi.csv`.

### Key design decisions

- **`irs_aggregate_values.csv` is not deduplicated** — intentionally preserves cross-table redundancy so minor differences are visible and auditable.
- **Deduplication happens in stage 3** — `Value` excluded from dedup key so ±1 integer differences between tables don't create duplicate rows. Tab11 is authoritative.
- **New `soi.csv` corrects 8 off-by-one errors** from the old R pipeline.
- **IRS Excel files remain in the repo** as ground truth; extracted CSVs committed for reproducibility.

### Tests

79 tests in `tests/test_national_targets.py` covering config sanity, extracted CSV shape and known IRS totals, `irs_aggregate_values.csv`, and `soi.csv`.

## Test plan

- [ ] `python -m pytest tests/test_national_targets.py -v`
- [ ] `make format && make lint`
- [ ] Full pipeline: `extract_irs_to_csv.py --overwrite` → `build_targets.py` → `potential_targets_to_soi.py`
- [ ] `make data` with TAXYEAR=2021 and TAXYEAR=2022

---
*Generated with Claude Code*
