# PR: Add script to auto-update regression test expected values

## Summary

- Adds `tests/update_expected_values.py` to automate updating regression test expected values after intentional data changes (growth rates, targets, etc.)

## Motivation

When inputs change deliberately (e.g., fixing a growth rate), the weight fingerprint, tax expenditure, and imputed variable tests fail because their expected values no longer match. Currently these must be updated manually — tedious and error-prone. This script automates it.

## Usage

```bash
# Dry run — show computed values without changing anything:
python tests/update_expected_values.py

# Update all expected values:
python tests/update_expected_values.py --all

# Update selectively:
python tests/update_expected_values.py --weights
python tests/update_expected_values.py --taxexp
python tests/update_expected_values.py --imputed
```

## What it updates

- `test_weights.py` — weight distribution fingerprint (n, total, mean, sdev, percentiles, etc.)
- `expected_tax_exp_2022_data` — tax expenditure estimates
- `test_imputed_variables.py` — OBBBA deduction benefit expected values

## Test plan

- [ ] Run `python tests/update_expected_values.py` (dry run) — verify it prints current values
- [ ] Run `python tests/update_expected_values.py --all` then `make test` — verify all tests pass
- [ ] Confirm no changes to test files when data hasn't changed (idempotent)
