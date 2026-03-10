# Session Notes: Examination of new-cps-code and new-puf-code PRs

## Date: 2026-03-10

## Overview

The `new-cps-code` branch (and related `new-puf-code` work) removes the `policyengine_us` dependency entirely, replacing it with direct data processing. This is a major refactoring that makes the TMD pipeline self-contained.

### Key architectural changes

- Deleted `tmd/datasets/taxcalc_dataset.py` (PE bridge module)
- New `create_tc_cps(taxyear)` in `cps.py` — builds CPS Tax-Calculator dataset directly from raw Census CSV data
- New `create_tc_puf(taxyear)` in `puf.py` — builds PUF Tax-Calculator dataset directly
- `create_tmd_2021()` became `create_tmd_dataframe(taxyear)` — parameterized by year
- Removed from `setup.py`: `policyengine_us`, `tables`, `behresp`, `tensorboard`, `jupyter-book`
- Python version bumped from `>=3.10,<3.13` to `>=3.11,<3.14`
- Net result: 18 commits, -576 lines

---

## Changes That Produce Different Results

### 1. Filing status determination (MARS) for CPS records

**Most important change.** MARS affects everything downstream — nonfiler thresholds, CTC, EITC, standard deduction, tax rates.

**Master (PE):** `sim.calculate("filing_status")` — complex rules engine.
**New code** (`cps.py:258-290`, `_derive_filing_status()`): Direct derivation from CPS `A_MARITL` field:
- `has_spouse` in tax unit → MARS=2 (joint)
- widowed (A_MARITL=4) + dependents + no spouse → MARS=5 (surviving spouse)
- divorced/separated/never-married + dependents → MARS=4 (HoH)
- everything else → MARS=1 (single)
- **Never assigns MARS=3 (MFS)**

#### MARS distribution for all 78,913 CPS tax units (before filtering)

| Filing status         | Master (PE) | New-cps-code |
|-----------------------|-------------|--------------|
| Single (MARS=1)       | 38,910      | 40,905       |
| MFJ (MARS=2)          | 31,627      | 29,850       |
| MFS (MARS=3)          | 2,067       | 0            |
| HoH (MARS=4)          | 6,010       | 7,560        |
| Surv. Spouse (MARS=5) | 299         | 598          |

#### Key reclassifications

**PE MFS → new code (2,067 records):**
- 1,979 of 2,067 are separated (A_MARITL=6). PE classified separated people as MFS. New code treats separated as "unmarried" → Single (1,214) or HoH (814) depending on dependents.

**PE MFJ → new Single (~1,764 records):**
- Heads whose A_MARITL says "married" (codes 1, 2, 3) but no spouse person present in the tax unit. PE classified by legal status; new code requires an actual spouse in the unit.

**PE HoH → new Surv. Spouse (~598 records):**
- Widowed heads with dependents. New code assigns all widowed+deps to MARS=5. PE was more restrictive (IRS surviving-spouse status requires death within prior 2 years).

#### Notable issue: A_MARITL=3 (married, spouse absent) with dependents

357 records are legally married, have dependents, but no spouse in the tax unit. New code assigns MARS=1 (Single) because maritl=3 isn't in the "unmarried" set [4,5,6,7] so they don't qualify for HoH. This is arguably wrong — these could be MFS or HoH depending on circumstances.

### 2. Nonfiler determination

**Master:** `sim.calculate("tax_unit_is_filer", period=2022)` — PE's internal filer model.
**New code** (`cps.py:617-647`): Explicit 2022 IRS gross income filing thresholds applied to sum of 12 income variables.

The new approach is more mechanical. PE's approach was more generous in calling people nonfilers (17,565 vs 12,269 CPS records in final TMD).

#### CPS records in final TMD by MARS

| Filing status         | Master  | New-cps-code | Difference |
|-----------------------|---------|--------------|------------|
| Single (MARS=1)       | 12,645  | 8,542        | -4,103     |
| MFJ (MARS=2)          | 2,901   | 1,836        | -1,065     |
| MFS (MARS=3)          | 462     | 0            | -462       |
| HoH (MARS=4)          | 1,463   | 1,689        | +226       |
| Surv. Spouse (MARS=5) | 94      | 202          | +108       |
| **Total CPS**         | **17,565** | **12,269** | **-5,296** |

PUF records are the same on both branches: 207,692.

**Concern:** Many people below IRS filing thresholds file nonetheless — to get refunds of withholding and to claim credits. The new approach may leave us with fewer people claiming credits than are eligible. This probably has minimal impact on overall tax liability estimates but could affect policy simulations involving credits (especially EITC). A placeholder issue should be opened to explore this regardless of which approach is kept.

#### Zero-weight CPS tax units

The new code explicitly drops tax units with zero `s006` weight (`cps.py:613-615`). Of 78,913 CPS tax units, 23,788 have zero weight (head's `A_FNLWGT` = 0). Master's PE-based pipeline handled these differently (PE may have filtered them internally or they may have been classified as filers and excluded that way).

### 3. EITC calculation (+17.5% in tax expenditure estimate)

EITC jumped from 77.9B to 91.5B. This is driven by a **PUF-side code change**, not the CPS nonfiler changes.

**Master:** `EIC` = raw PUF value only.
**New code** (`puf.py:421-426`, commit 66fe7da): `np.where(eic_raw > 0, eic_raw, eic_age_elig)` — records with EIC=0 in the raw PUF get a fallback count of dependents under 19. This fills in EIC eligibility for PUF records that previously showed zero qualifying children despite having young dependents.

**CPS side** (`cps.py:518-522`): Counts EITC children as age < 19 OR (age < 24 AND full-time student). PE may have used different logic.

### 4. Social Security partial taxability (-12.2%)

Dropped from 57.5B to 50.5B. Likely traces to filing-status shift. SS taxability depends on "combined income" thresholds that differ by MARS: $25,000 for singles vs $32,000 for joint. Reclassifying units from joint to single/HoH lowers the threshold → more SS taxed → less tax expenditure from the partial exclusion.

### 5. CTC increase (+2.0%)

Similar to EITC: driven by `f2441` fallback logic in `puf.py:428-430`. Records with `f2441_raw=0` now get a fallback count of dependents under 13.

### 6. Pension contribution imputation

**Master:** PE's `traditional_401k_contributions + traditional_403b_contributions`.
**New code** (`pension_contributions.py`): Allocates raw CPS `RETCB_VAL` directly — self-employment pension first, then 401(k) up to IRS limits.

**Naming issue:** The variable `trad_401k` (in `cps.py:469` and `pension_contributions.py:52`) is misleading. CPS `RETCB_VAL` captures total pretax contributions to any employer-sponsored DC plan (401(k), 403(b), 457(b), thrift savings). The limits for all these plan types are identical, so the cap is correct — but the name should be something like `pretax_retirement_contrib` or `employer_plan_contrib`.

### 7. Other changes

- `test_area_weights.py`: e00200 tolerance loosened to 0.008
- `test_misc.py`: Population target changed from 334.18M to 331.894M (2021 Census estimate)
- Weight distribution shifted: n=225,256→219,961, mean weight 816→842
- Imputed variable distributions (overtime, tip income) shifted 5-15%
- RNG seeds centralized in `imputation_assumptions.py`
- Dead code removed from `soi_replication.py`

---

## Test expectation changes

| Tax expenditure item          | Master   | New      | Change  |
|-------------------------------|----------|----------|---------|
| paytax                        | 1,382.9B | 1,389.3B | +0.5%   |
| iitax                         | 2,245.3B | 2,246.0B | +0.03%  |
| ctc                           | 129.5B   | 132.1B   | +2.0%   |
| **eitc**                      | **77.9B** | **91.5B** | **+17.5%** |
| **ss_partial_taxability**     | **57.5B** | **50.5B** | **-12.2%** |
| niit                          | -44.4    | -44.4    | 0       |
| cgqd_tax_preference           | 174.6    | 174.6    | 0       |
| qbid                          | 52.7     | 52.7     | 0       |

---

## Open questions / suggestions

1. **Nonfiler determination:** Consider opening a placeholder issue to explore whether IRS-threshold approach misses credit-eligible filers
2. **MARS=3 (MFS):** New code never assigns it — is that acceptable? PE assigned 2,067 CPS units as MFS (mostly separated people)
3. **MARS=5 (surviving spouse):** New code assigns all widowed+deps regardless of when spouse died; IRS requires death within prior 2 years
4. **A_MARITL=3 + deps:** 357 married-spouse-absent records with dependents assigned MARS=1; arguably should be HoH or MFS
5. **`trad_401k` naming:** Should be renamed to reflect it covers all DC plan types, not just 401(k)
