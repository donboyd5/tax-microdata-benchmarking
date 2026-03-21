# PR: Use Census-based growth rate for SALT deductions during PUF uprating

## Summary

- Replaces the 2%/year growth rate for SALT deductions (E18400, E18500) with a Census-based 4.7%/year rate
- Keeps the existing 2%/year rate for non-SALT itemized deductions (medical, mortgage interest, charitable)
- Updates regression test expected values to match new output

## Motivation

The 2%/year rate significantly understates SALT growth from 2015 to 2022. Census Bureau data on actual state and local tax collections shows much faster growth:

| Tax type | 2015 | 2022 | Growth |
|----------|-----:|-----:|-------:|
| Property taxes | $484.3B | $649.0B | +34.0% |
| General sales taxes | $368.0B | $557.2B | +51.3% |
| Individual income taxes | $368.9B | $600.6B | +62.8% |
| **SALT proxy total** | **$1,221.2B** | **$1,806.8B** | **+48.0%** |

Per-return (dividing out returns growth): **+38.0%**, implying ~4.7%/year.

The current 2%/year gives only +14.9% per-return over 7 years — understating SALT growth by roughly a factor of 2.5.

Source: U.S. Census Bureau, Annual Survey of State and Local Government Finances, Table 1.
- 2015: https://www2.census.gov/programs-surveys/gov-finances/tables/2015/summary-tables/15slsstab1a.xlsx
- 2022: https://www2.census.gov/programs-surveys/gov-finances/tables/2022/22slsstab1.xlsx

## Changes

- `tmd/imputation_assumptions.py`: Add `SALT_GROW_RATE = 0.047`, rename comment on `ITMDED_GROW_RATE` to clarify it applies to non-SALT deductions only
- `tmd/datasets/uprate_puf.py`: Apply `SALT_GROW_RATE` to E18400 and E18500, keep `ITMDED_GROW_RATE` for medical, mortgage interest, and charitable
- `tests/`: Updated expected values via `python tests/update_expected_values.py --all`

## Caveats

- Census data includes taxes paid by businesses, not just individuals. The SALT deduction is for individual taxes only. This is the best available proxy.
- A single rate is applied to both E18400 (income/sales tax) and E18500 (property tax). In practice, property taxes grew slower (+34%) than income/sales taxes (+57%). Separate rates could be a future refinement.

## Known test issue

`test_population` will fail — population is 335.12M vs the expected 334.00M (0.33% off, tolerance is 0.1%). This is a correctness test against Census population data, not a regression test. The SALT growth change affects reweighting, which shifts total population slightly. This requires a decision on whether to widen the tolerance or investigate further.

## Test plan

- [x] `make format` passes
- [x] `make lint` passes
- [x] `make test` — 3 regression tests pass after expected value updates; `test_population` fails (see above)