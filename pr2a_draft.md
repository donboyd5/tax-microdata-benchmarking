# PR #2a: Parameterize TAXYEAR

**Title:** Parameterize TAXYEAR: single source of truth for tax year across pipeline

**Body:**

## Summary

Goal: Make it easier to switch the national target year from 2021 to 2022 or later years. This requires eliminating hardcoded instances of 2021, replacing with a parameter value, TAXYEAR.

As implemented, this PR puts in place the necessary code to do this. With the default TAXYEAR=2021 it produces identical results to the master branch. If run with TAXYEAR=2022, it will run and create a file with 2022 IRS targets but that file is **NOT READY** for use. We have several major changes yet to come, including (1) updating the CPS from 2021 to 2022, (2) calibrating growth factors for 2022→2023 (and aligning upstream `puf_growfactors.csv`), (3) generalizing the function that creates the tmd file (see note to @martinholmer below), (4) fixing Tax-Calculator's `tmd_constructor()` which hardcodes `start_year=2021`, and (5) updating expected values in tests to reflect what we expect in 2022. These changes will be addressed in future pull requests.

- Introduce `TAXYEAR = 2021` in `imputation_assumptions.py` as the single source of truth
  for the tax year used throughout the pipeline
- Eliminate all hardcoded `2021` values and duplicate `TAX_YEAR` aliases across 13 files
- Bypass Tax-Calculator's `tmd_constructor()` (which hardcodes `start_year=2021`) in both
  the pipeline and tests, constructing `tc.Records()` directly with `start_year=TAXYEAR`
- Guard @martinholmer's 2022 growth-factor calibration adjustments so they only apply when
  growing FROM an earlier year TO 2022 (not when 2022 IS the base year)

**Verified:** With `TAXYEAR=2021`, `make clean && make data` passes all 51 tests identically
to master. With `TAXYEAR=2022`, the full pipeline completes without crashes — 5 tests fail
on value mismatches (expected, since test fingerprints are calibrated for 2021 data).

This PR does NOT change the default tax year. It stays at 2021. The goal is to make the
eventual switch to 2022 a one-line change.

## Files changed (12 modified, 1 added)

| File | Change |
|------|--------|
| `tmd/imputation_assumptions.py` | Added `TAXYEAR = 2021` (single source of truth) |
| `tmd/create_taxcalc_input_variables.py` | Import `TAXYEAR` instead of defining it locally |
| `tmd/utils/reweight.py` | Removed `TAX_YEAR` alias, use `TAXYEAR` directly |
| `tmd/utils/reweight_clarabel.py` | Import `TAXYEAR` from imputation_assumptions |
| `tmd/datasets/tmd.py` | Replaced 5 hardcoded `2021` values with `TAXYEAR` |
| `tmd/create_taxcalc_cached_files.py` | Bypass `tmd_constructor()`, construct `tc.Records()` with `start_year=TAXYEAR` |
| `tmd/create_taxcalc_growth_factors.py` | Guard 2022 calibration adjustments with `if FIRST_YEAR < 2022:` |
| `tmd/utils/taxcalc_utils.py` | Changed `assert input_data_year == 2021` to `assert input_data_year == TAXYEAR` |
| `tests/conftest.py` | Added `create_tmd_records()` helper (same bypass as pipeline) |
| `tests/__init__.py` | New empty file (makes `tests` importable for conftest import) |
| `tests/test_misc.py` | Use `create_tmd_records()` instead of `tmd_constructor()` |
| `tests/test_area_weights.py` | Same |
| `tests/test_imputed_variables.py` | Same |

## Known issue: `create_tmd_2021()` needs to be generalized

**@martinholmer** — `tmd/datasets/tmd.py` has a function called `create_tmd_2021()` that
hardcodes `CPS_2021` and `PUF_2021` data classes. This PR parameterized the *tax year
arithmetic* inside the function (Tax-Calculator runs, reweighting, growth factors all use
`TAXYEAR`), but the function still always loads **2021 CPS microdata** regardless of
`TAXYEAR`. (The PUF side is always the same 2015 base file — `puf_2015.csv` — so
it's not affected in the same way.)

When `TAXYEAR=2022`, the pipeline currently "works" by using 2021 CPS nonfilers, aging
everything forward via growth factors, and reweighting to 2022 IRS targets — but it's
not using actual 2022 CPS data. The function (originally written by Nikhil Woodruff, then
substantially extended by Martin) was designed as a one-shot for 2021 and needs to be
refactored to accept the data year as a parameter, or dispatch to year-specific dataset
classes.

This is the single biggest piece of work remaining for a true 2022 switch. We'd welcome
Martin's input on the right approach — e.g., should `create_tmd_2021()` become
`create_tmd(taxyear, cpsyear)`, or should there be separate `create_tmd_2022()`, or
something else?

## What else is needed before we can use 2022 data

Switching TAXYEAR to 2022 is now a one-line change, but several other pieces must be
in place first:

### Must-do before switching to 2022

1. **CPS 2022 dataset classes and `create_tmd_2021()` generalization** — `cps.py` has
   the 2022 CPS URL but no `RawCPS_2022`, `CPS_2022`, or `create_cps_2022()`. These
   must be added, and `create_tmd_2021()` in `tmd.py` must be refactored to use
   year-appropriate CPS/PUF classes (see "Known issue" above).
   *(Planned as PR #3 in the 5-PR strategy.)*

2. **2022 growth factor calibration** — Martin's hardcoded adjustments calibrate the
   2022 row of `tmd_growfactors.csv` when growing from 2021. When TAXYEAR=2022, the
   2022 row is the baseline (all ones) and reweighting handles calibration via SOI
   targets. However, growing FROM 2022 to 2023 (for tax expenditure simulations) will
   need analogous calibration adjustments for the 2023 row. Martin should be consulted
   on methodology and data sources. *(Separate issue to create.)*

3. **Tax-Calculator `tmd_constructor()` hardcodes `start_year=2021`** — We work around
   this by constructing `tc.Records()` directly. Long-term, Tax-Calculator should accept
   a `start_year` parameter in `tmd_constructor()`. *(Separate issue to create in
   tax-calculator repo.)*

4. **Test fingerprint values** — Tests like `test_income_tax` and `test_variable_totals`
   have expected values calibrated for 2021. These need updating for 2022 output.
   *(Planned as PR #4.)*

5. **Upstream `puf_growfactors.csv` alignment** — This file is from Tax-Calculator
   (calibrated for taxdata's PUF). For TMD's combined PUF+CPS data, the standard growth
   factors may not perfectly match IRS published aggregates. The current 2022 adjustments
   address this for 2021→2022; similar work is needed for 2022→2023.

### 5-PR strategy overview

| PR | Description | Status |
|----|-------------|--------|
| PR #1 (PR #424) | Infrastructure: soi.csv with 2022 targets, converter, mapping | **Merged** |
| **PR #2a (this PR)** | **Parameterize TAXYEAR (keep default=2021)** | **Ready** |
| PR #2b | Flip TAXYEAR default to 2022 | Depends on #2a + #3 |
| PR #3 | CPS 2022 dataset classes | Parallel with #2a |
| PR #4 | Update test fingerprints for 2022 | Depends on #2b |

## Test plan

- [x] `make clean && make data` with TAXYEAR=2021: 51 passed, 3 skipped (identical to master)
- [x] `make clean && make data` with TAXYEAR=2022: pipeline completes, 5 tests fail on
      expected value mismatches (no crashes)
- [x] `make format` (black): no changes needed
- [x] `pycodestyle`: zero issues
- [x] `pylint`: no new warnings (all pre-existing)
