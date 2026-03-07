**Read `repo_conventions_session_notes.md` first.**

# CPS Exploration Notes
*Branch: `cps-exploration` (based on `pr2a-parameterize-taxyear`)*
*Created: 2026-03-06*

---

## Goal

Understand how CPS (Current Population Survey) data is used in this project, what variables matter, and what's needed to support CPS 2022 (income year 2022, March 2023 survey).

---

## CPS Naming Convention (Critical)

The CPS Annual Social and Economic Supplement (ASEC) has a confusing naming pattern:

| Income year | Survey administered | Census file name | Code's `time_period` |
|-------------|--------------------|-----------------|-----------------------|
| 2020 | March 2021 | asecpub21csv.zip | 2020 |
| **2021** | **March 2022** | **asecpub22csv.zip** | **2021** |
| **2022** | **March 2023** | **asecpub23csv.zip** | **2022** |

The code uses `time_period` = **income year** (not file year). The file year is `time_period + 1`.

---

## How CPS Is Used in This Project

### Overview

CPS provides the **nonfiler population**. The final TMD file = PUF filers (all records) + CPS nonfilers (subset). The pipeline:

```
Census CPS ASEC ZIP
  → RawCPS: download, extract person/family/household CSVs into HDF5
  → CPS: transform to PolicyEngine variable names (h5py arrays)
  → create_tc_dataset(): convert to Tax-Calculator variable format
  → create_tmd_2021(): identify nonfilers, combine with PUF, reweight
  → tmd.csv.gz
```

### Stage 1: Download & Extract — `RawCPS` class (cps.py:140-265)

Downloads the ZIP from Census, reads three CSVs (`pppub{YY}.csv`, `ffpub{YY}.csv`, `hhpub{YY}.csv`), and stores five HDF5 tables: person, family, household, tax_unit, spm_unit.

**Year-specific classes:**
- `RawCPS_2021` (cps.py:285-289) — currently the only concrete implementation
- A `RawCPS_2022` class is needed for PR #3

### Stage 2: Transform — `CPS` class (cps.py:292-327)

Calls helper functions to create PolicyEngine-format h5py arrays:
- `add_id_variables()` — person/family/household/tax_unit/spm_unit IDs and weights
- `add_personal_variables()` — demographics (age, sex, race, disability, etc.)
- `add_personal_income_variables()` — all income with tax-concept mappings
- `add_previous_year_income()` — links to prior year via PERIDNUM (currently unused)
- `add_spm_variables()` — SPM benefit and expense variables
- `add_household_variables()` — geography (state, county, NYC)

**Year-specific classes:**
- `CPS_2021` (cps.py:785-791) — uses `RawCPS_2021`, `time_period=2021`
- A `CPS_2022` class is needed for PR #3

### Stage 3: Convert to Tax-Calculator Format — `create_tc_dataset()` (taxcalc_dataset.py:17-221)

Maps 60+ PolicyEngine variables to Tax-Calculator variable names. Aggregates person-level to tax-unit level. Splits income by head/spouse. Sets `data_source=0` for CPS, `data_source=1` for PUF.

### Stage 4: Combine PUF + CPS Nonfilers — `create_tmd_2021()` (tmd.py:16-110)

1. Identify CPS nonfilers using **2022 filing rules** (avoids 2021 COVID anomalies)
2. Keep only CPS nonfiler records
3. Concatenate PUF (all) + CPS (nonfilers only)
4. Drop CPS records with positive income tax (shouldn't happen for nonfilers)
5. Scale CPS weights by `CPS_WEIGHTS_SCALE = 0.5806`
6. Reweight combined file against SOI targets

---

## CPS Variables Used

### Person-Level Variables Read from Census CSVs (PERSON_COLUMNS, 63 vars)

**IDs & Weights:**
- `PH_SEQ`, `PF_SEQ`, `P_SEQ` — household/family/person sequence numbers
- `TAX_ID`, `SPM_ID` — tax unit and SPM unit identifiers
- `A_FNLWGT` — final person weight
- `A_LINENO`, `A_SPOUSE` — line number, spouse line number
- `PERIDNUM` — person ID for longitudinal matching (used in previous-year linking)

**Demographics:**
- `A_AGE` — age (80=80-84, 85=85+; code randomizes 80 to 80-84)
- `A_SEX` — sex (1=male, 2=female)
- `A_MARITL` — marital status
- `PRDTRACE`, `PRDTHSP` — race, Hispanic origin
- `A_HSCOL` — school enrollment
- `PEDISEYE`, `PEDISDRS`, `PEDISEAR`, `PEDISOUT`, `PEDISPHY`, `PEDISREM` — disability flags
- `PEPAR1`, `PEPAR2` — parent line numbers (for counting children)
- `MRK` — marketplace health coverage

**Income Variables:**

| CPS Variable | PolicyEngine Variable(s) | Mapping |
|-------------|--------------------------|---------|
| `WSAL_VAL` | `employment_income` | Direct |
| `INT_VAL` | `taxable_interest_income` + `tax_exempt_interest_income` | Split 68%/32% (SOI 2020) |
| `DIV_VAL` | `qualified_dividend_income` + `non_qualified_dividend_income` | Split 44.8%/55.2% (SOI 2018) |
| `SEMP_VAL` | `self_employment_income` | Direct |
| `FRSE_VAL` | `farm_income` | Direct |
| `RNT_VAL` | `rental_income` | Direct |
| `SS_VAL` | `social_security_retirement` or `social_security_disability` | Split by age >= 62 |
| `UC_VAL` | `unemployment_compensation` | Direct |
| `PNSN_VAL` + `ANN_VAL` | `taxable_private_pension_income` + `tax_exempt_private_pension_income` | 100% taxable (arbitrary) |
| `CAP_VAL` | `long_term_capital_gains` + `short_term_capital_gains` | Split 88%/12% (SOI 2012) |
| `RETCB_VAL` | Multiple retirement contribution types | Allocated by rules (see below) |
| `OI_VAL` (code 20) | `alimony_income` | Filtered by `OI_OFF` |
| `OI_VAL` (code 12) | `strike_benefits` | Filtered by `OI_OFF` |
| `CSP_VAL` | `child_support_received` | Direct |
| `PAW_VAL` | `tanf_reported` | Assumed TANF (could be Gen. Assistance) |
| `SSI_VAL` | `ssi_reported` | Direct |
| `VET_VAL` | `veterans_benefits` | Direct |
| `WC_VAL` | `workers_compensation` | Direct |
| `DIS_VAL1`, `DIS_VAL2`, `DIS_SC1`, `DIS_SC2` | `disability_benefits` | Excluding workers comp |
| `DST_SC1/2`, `DST_VAL1/2` (+ `_YNG`) | Retirement distributions by type | Codes 1-7 map to account types |

**Expense Variables:**
- `CHSP_VAL` → `child_support_expense`
- `PHIP_VAL` → `health_insurance_premiums`
- `MOOP` → `medical_out_of_pocket_expenses`

**Work Variables:**
- `HRSWK`, `WKSWORK` → `weekly_hours_worked`
- `I_ERNVAL`, `I_SEVAL` — imputation flags (used for previous-year linking)

### Tax Unit Variables (TAX_UNIT_COLUMNS, 10 vars)

`ACTC_CRD`, `AGI`, `CTC_CRD`, `EIT_CRED`, `FEDTAX_AC`, `FEDTAX_BC`, `MARG_TAX`, `STATETAX_A`, `STATETAX_B`, `TAX_INC`

These are aggregated by `TAX_ID` from the person file. Used in the `create_tc_dataset()` conversion.

### SPM Unit Variables (SPM_UNIT_COLUMNS, 39 vars)

Supplemental Poverty Measure data — government benefits, expenses, poverty thresholds. Key ones:
- `SPM_SNAPSUB` → SNAP benefits
- `SPM_CAPHOUSESUB` → housing subsidy
- `SPM_SCHLUNCH` → school meals
- `SPM_ENGVAL` → energy subsidy
- `SPM_WICVAL` → WIC
- `SPM_BBSUBVAL` → broadband subsidy (**new in 2021, not in 2020 and earlier**)
- `SPM_FICA`, `SPM_FEDTAX`, `SPM_STTAX` — tax amounts
- `SPM_POVTHRESHOLD`, `SPM_RESOURCES` — poverty measures

### Household Variables

- `GESTFIPS` → `state_fips`
- `GTCO` → `county_fips`
- `HSUP_WGT` → `household_weight`

### Family Variables

- `FSUP_WGT` → `family_weight`

---

## Imputation Parameters (imputation_parameters.yaml)

These fixed fractions split CPS totals into tax-relevant components:

| Parameter | Value | Source |
|-----------|-------|--------|
| `taxable_interest_fraction` | 0.680 | SOI 2020 |
| `qualified_dividend_fraction` | 0.448 | SOI 2018 |
| `taxable_pension_fraction` | 1.0 | No data (arbitrary) |
| `long_term_capgain_fraction` | 0.880 | SOI 2012 |
| `taxable_401k_distribution_fraction` | 1.0 | Arbitrary |
| `taxable_403b_distribution_fraction` | 1.0 | Arbitrary |
| `taxable_ira_distribution_fraction` | 1.0 | Arbitrary |
| `taxable_sep_distribution_fraction` | 1.0 | Arbitrary |

These are **not year-parameterized** — same fractions used regardless of `TAXYEAR`.

---

## Retirement Contribution Allocation (cps.py:595-646)

CPS reports a single `RETCB_VAL` (total retirement contributions). The code allocates:
1. If self-employed → all to `self_employed_pension_contributions`
2. If wage earner → sequential allocation:
   - Traditional 401k (up to $20,500 + $6,500 catch-up if age 50+)
   - Roth 401k (same limit)
   - Traditional IRA (up to $6,000 + $1,000 catch-up)
   - Roth IRA (remainder of IRA limit)

**Note:** Limits are hardcoded as 2022 values. For 2023 (income year 2022), limits increased:
- 401k: $22,500 (+$7,500 catch-up)
- IRA: $6,500 (+$1,000 catch-up)
These should be parameterized by year.

---

## Year-Specific Differences in CPS Code

### A. SPM_BBSUBVAL (broadband subsidy)
- **Not available in 2020 and earlier** CPS files
- Code at cps.py:185-188 conditionally excludes for `time_period <= 2020`
- Available in 2021 and 2022

### B. File path special case (2018/2019)
- 2018 CPS files nested under subdirectory `cpspb/asec/prod/data/2019/`
- All other years at ZIP root (cps.py:222-227)

### C. Nonfiler identification
- Uses **2022 filing rules** even for 2021 data (tmd.py:25-28)
- Avoids COVID-related anomalies in 2021 rules (expanded CTC, etc.)
- When moving to 2022, should this use 2023 filing rules?

### D. CPS_WEIGHTS_SCALE = 0.5806
- Scales CPS weights after combining with PUF
- Defined in `imputation_assumptions.py`
- **May need recalibration for 2022** — depends on CPS sample size and nonfiler population

---

## Known Differences Between 2022 and 2023 ASEC Files (Census Documentation)

Based on Census technical documentation (cpsmar23.pdf):

- **One new variable**: `UH_STTAXREB_A1` — state tax rebate amount
- **INCRETIR fix**: Not-in-universe responses were incorrectly coded as "0" rather than "99999999" for NIU in all years after 2018. Corrected in 2023 release.
- **No changes to the core income variables** used by this project (WSAL_VAL, INT_VAL, DIV_VAL, etc.)

The 63 PERSON_COLUMNS, 10 TAX_UNIT_COLUMNS, and 39 SPM_UNIT_COLUMNS used by the code **should all still be present in the 2023 ASEC (income year 2022)**.

---

## What's Needed for PR #3: CPS 2022 Support

Based on the 5-PR strategy from national_targets_session_notes.md:

### A. New classes (in cps.py):

```python
class RawCPS_2022(RawCPS):
    time_period = 2022
    name = "raw_cps_2022"
    label = "Raw CPS 2022"
    file_path = STORAGE_FOLDER / "input" / "raw_cps_2022.h5"

class CPS_2022(CPS):
    name = "cps_2022"
    label = "CPS 2022"
    raw_cps = RawCPS_2022
    file_path = STORAGE_FOLDER / "output" / "cps_2022.h5"
    time_period = 2022
```

### B. Update tmd.py to use CPS_2022 when TAXYEAR=2022

Currently hardcoded to `CPS_2021` / `create_cps_2021()`. Need to select by year.

### C. Retirement contribution limits

Hardcoded 2022 limits (for income year 2021). For income year 2022, limits are different. Should be parameterized.

### D. Nonfiler identification year

Currently uses period=2022 for filing rules. When TAXYEAR=2022, should use period=2023?

### E. CPS_WEIGHTS_SCALE

May need recalibration for 2022 data.

### F. Imputation parameters

The split fractions (interest, dividends, capital gains) are based on specific SOI years. May want to update for 2022. Low priority — fractions are fairly stable.

---

## CPS Documentation Links

- [CPS ASEC Data Downloads (all years)](https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html)
- [2023 ASEC Data (income year 2022)](https://www.census.gov/data/datasets/2023/demo/cps/cps-asec-2023.html)
- [Complete CPS Technical Documentation](https://www.census.gov/programs-surveys/cps/technical-documentation/complete.html)
- Technical doc PDFs (not readable via web fetch, need local download):
  - `https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar22.pdf` (2021 income year)
  - `https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar23.pdf` (2022 income year)
- [ASEC 2023 Data Dictionary (PDF)](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf)

---

## Open Questions

1. **CPS_WEIGHTS_SCALE for 2022** — is 0.5806 still appropriate? How was it derived?
2. **Nonfiler identification period** — when TAXYEAR=2022, use 2023 filing rules?
3. **Retirement contribution limits** — parameterize by year?
4. **Imputation fractions** — update for more recent SOI data?
5. **INCRETIR coding fix** — does this affect any variables used in the pipeline?
6. **UH_STTAXREB_A1** (new in 2023 ASEC) — relevant for state tax modeling?

---

## Resume Instructions

When resuming this session:
1. Read `repo_conventions_session_notes.md` first
2. Currently on **`cps-exploration`** branch (based on `pr2a-parameterize-taxyear`)
3. The primary file to understand is `tmd/datasets/cps.py` (799 lines)
4. Key companion files: `tmd/datasets/tmd.py`, `tmd/datasets/taxcalc_dataset.py`, `tmd/imputation_assumptions.py`
5. PR #3 goal: add `RawCPS_2022` + `CPS_2022` classes and make `tmd.py` year-aware
