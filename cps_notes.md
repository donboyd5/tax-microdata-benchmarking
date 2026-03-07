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

## Session Update: 2026-03-07

### CPS 2021 Weighted Counts (Income Year 2021, March 2022 Survey)

| Level | Records | Weighted Total |
|-------|--------:|---------------:|
| Persons | 152,732 | 327,809,775 |
| Households | 59,148 | 131,314,715 |
| Families | 66,529 | 148,631,518 |
| Tax units (all) | 78,913 | 175,310,313 |
| Tax units (filers) | 61,348 | 136,852,192 |
| Tax units (nonfilers) | 17,565 | 38,458,120 |

Weights used: `A_FNLWGT/100` for persons, `HSUP_WGT/100` for households, `FSUP_WGT/100` for families.
Tax unit weight = family weight of containing family (first person in tax unit gets family weight).
Filer/nonfiler split uses PolicyEngine `tax_unit_is_filer` with `period=2022` rules.

### CPS Weight Flow in the Pipeline

```
All CPS tax units:        175.3M weighted (78,913 records)
  → Filter to nonfilers:   38.5M weighted (17,565 records)
  → × CPS_WEIGHTS_SCALE:   22.3M weighted (0.5806 × 38.5M)
  → Drop positive iitax:   (removes a few records)
  → Reweight with PUF:     22.3M in final TMD

Final TMD:
  PUF records: 207,692 records, 161.6M weighted
  CPS records:  17,564 records,  22.3M weighted
  Total:       225,256 records, 183.9M weighted
```

### Q&A: How CPS Interacts with PUF

**Q: Are any CPS variables added to PUF records?**

Indirectly, one case. `pension_contributions.py` (line 12) loads CPS_2021, extracts
`employment_income`, `traditional_401k_contributions`, and `traditional_403b_contributions`,
trains a RandomForest model on CPS data, then uses it to *predict* pension contributions
(`pencon_p`, `pencon_s`) for PUF records. This happens inside `create_tc_dataset()`
(`taxcalc_dataset.py:171-184`). PUF lacks pretax pension data, so CPS is the training source.
No other CPS→PUF transfer occurs — the datasets are simply concatenated (`tmd.py:32`).

**Q: Are CPS records reweighted, other than the .58 adjustment?**

Yes. The .5806 is applied BEFORE reweighting (`tmd.py:45-46`). Then all records (PUF + CPS
together) enter the Clarabel QP optimizer. The reweighter first prescales all weights to match
the SOI filer count exactly (`reweight_clarabel.py:389-408`), then the optimizer adjusts
individual weights to hit all 550 SOI targets. So CPS records get three adjustments:
(1) CPS_WEIGHTS_SCALE=0.5806, (2) prescale to SOI filer count, (3) per-record optimizer weights.

**Q: Why the .58 adjustment? Documentation?**

It scales 38.5M nonfiler tax units down to ~22.3M. Martin Holmer introduced it in two commits
on August 30, 2024:
- PR #175, commit `21e1ee46`: "Scale CPS record weights" — initial value 0.6248
- PR #176, commit `04c5fd0a`: "Rescale CPS weights" — refined to 0.5806 (same day)

No derivation documented in commit messages, code comments, or PolicyEngine docs. The only
comment is `# used to scale CPS-subsample population` (`imputation_assumptions.py:16`).
It appears to be an empirically calibrated parameter. The rationale for cutting nonfiler
tax units almost in half is unclear — needs investigation.

**Q: What code is used from PolicyEngine?**

Three packages:
- `policyengine_core.data.Dataset` — base class for RawCPS, CPS, and PUF dataset classes.
  Provides HDF5 storage, generate()/load() pattern, data format handling (TABLES vs ARRAYS).
- `policyengine_us.Microsimulation` — tax calculation engine. Used to:
  (A) identify nonfilers via `tax_unit_is_filer` (`tmd.py:27-28`),
  (B) extract/aggregate all variables in `create_tc_dataset()` (`taxcalc_dataset.py:18`),
  (C) train pension contribution imputation (`pension_contributions.py:12`),
  (D) convert PE variables to SOI comparison format (`soi_replication.py`).
- `policyengine_us.system` — variable metadata (entity keys, variable names).
  Used in `taxcalc_dataset.py:10` and `puf.py:6`.

**Q: Nonfiler identification period for 2022?**

Decision: use `period=2022` for both TAXYEAR=2021 and TAXYEAR=2022. The current code already
uses 2022 filer rules for 2021 data (to avoid COVID anomalies). For minimalism, keep the
same period for both years.

### PUF Also Needs a 2022 Class

`PUF_2021` has `time_period=2021`, which controls `uprate_puf(puf, 2015, self.time_period)`.
Without a `PUF_2022` (time_period=2022), PUF data would be uprated to 2021 instead of 2022.
`soi.csv` already has 2022 data (from PR #424), so growth factors will work.
`create_pe_puf_2022()` reads the same `puf_2015.csv` + `demographics_2015.csv` as 2021 —
the only difference is the uprating target year.

### Nonfiler Tax Unit Estimates from External Sources

How many nonfiler tax units are there in the US? The CPS says 38.5M, TMD scales to 22.3M.
External sources (all figures for ~2021 unless noted):

**Tax-unit-based estimates (apples-to-apples):**

| Source | Total Tax Units | Filers | Nonfilers | Notes |
|--------|:-:|:-:|:-:|-------|
| Treasury OTA (TP-8, May 2022) | 183M | ~151M | 32M (18%) | Distribution model; best single benchmark |
| Tax Policy Center | ~177M | — | — | Microsimulation model baseline |
| CPS ASEC (raw, this project) | 175M | 136.9M | 38.5M | Census tax model; known to overcount nonfilers |
| **TMD (after .5806 scale)** | **183.9M** | **161.6M** | **22.3M** | PUF filers + scaled CPS nonfilers |
| IRS SOI | — | 160.8M | — | Actual returns filed |

**Individual-based estimates (different unit — not directly comparable to above):**

| Source | Estimate | Unit | Notes |
|--------|----------|------|-------|
| IRS Research (Langetieg et al.) | 10-15M | Individuals | *Required* nonfilers only (should file but don't) |
| CBPP (stimulus outreach) | 12M | Individuals | Nonfilers eligible for stimulus |
| Cilke/Treasury (~10% of pop.) | ~33M | Individuals | Rough approximation |

**Key observations:**
- CPS nonfiler count (38.5M tax units) is widely known to be an overcount due to income
  underreporting in survey data. Many CPS "nonfilers" actually have enough income to file.
- Treasury OTA's 32M nonfiler tax units is the most authoritative tax-unit estimate.
- TMD's 22.3M nonfilers is well below Treasury's 32M — the .5806 scale may be too aggressive.
- TMD's filer count (161.6M) matches IRS SOI (160.8M) well — the PUF side is calibrated.
- Treasury OTA says 151M filers vs IRS SOI 160.8M — a 10M gap. Possible definitional
  differences (OTA may exclude some returns, or use a different filing unit definition).
- The CPS filer count (136.9M) is far below IRS (160.8M) — the CPS significantly undercounts
  filers, likely because income underreporting causes some true filers to be classified as
  nonfilers in the PolicyEngine `tax_unit_is_filer` calculation.

**Sources:**
- Treasury OTA Technical Paper 8 (May 2022): https://home.treasury.gov/system/files/131/TP-8.pdf
- IRS nonfiler research: https://www.irs.gov/pub/irs-soi/17resconpayne.pdf
- Census CPS tax model: https://www.census.gov/topics/income-poverty/income/guidance/tax-model.html
- Cilke, Treasury WP-78: https://home.treasury.gov/system/files/131/WP-78.pdf

### Session Summary (2026-03-07)

**What we learned:**

1. **CPS structure**: The CPS 2021 has 152,732 person records (328M weighted), 59,148 households
   (131M), 66,529 families (149M), and 78,913 tax units (175M). Of those tax units, 61,348 are
   filers (137M weighted) and 17,565 are nonfilers (38.5M weighted).

2. **CPS role in TMD**: CPS provides only the nonfiler population. PUF provides all filers.
   The two are concatenated — no variable transfer except pension contribution imputation
   (CPS trains a model, PUF gets predictions). CPS nonfiler weights are scaled by .5806,
   then all records are reweighted together against SOI targets.

3. **The .5806 mystery**: Scales 38.5M nonfiler tax units to ~22.3M. Introduced by Martin Holmer
   (Aug 2024), no derivation documented. Treasury OTA estimates 32M nonfiler tax units for 2021,
   so TMD's 22.3M may be too low. The CPS also undercounts filers (137M vs IRS 160.8M) due to
   income underreporting — many true filers look like nonfilers in CPS data.

4. **PolicyEngine dependency**: Three packages provide the Dataset base class, the tax
   calculation engine (Microsimulation), and variable metadata. PolicyEngine determines
   filer/nonfiler status and extracts all variables from CPS/PUF into Tax-Calculator format.

5. **For CPS 2022 support**: Need new classes (RawCPS_2022, CPS_2022, PUF_2022) and
   TAXYEAR-aware selection in tmd.py and pension_contributions.py. PUF_2022 is critical —
   `time_period` controls uprating from 2015 base data. All year-specific parameters
   (CPS_WEIGHTS_SCALE, retirement limits, imputation fractions) kept unchanged for minimalism.

---

## Open Questions

1. **CPS_WEIGHTS_SCALE**: 0.5806 produces 22.3M nonfilers vs Treasury OTA's 32M. Too aggressive?
   How was it derived? Should it change for 2022?
2. **CPS filer undercount**: CPS identifies only 137M filers vs IRS 160.8M. Is this a known
   limitation of PolicyEngine's `tax_unit_is_filer`, or of CPS income underreporting, or both?
3. **Retirement contribution limits** — hardcoded as 2022 values. Parameterize by year?
4. **Imputation fractions** — update for more recent SOI data?
5. **INCRETIR coding fix** in 2023 ASEC — does this affect any variables used in the pipeline?
6. **UH_STTAXREB_A1** (new in 2023 ASEC) — relevant for state tax modeling?

---

## Resume Instructions

When resuming this session:
1. Read `repo_conventions_session_notes.md` first
2. Currently on **`cps-exploration`** branch (based on `pr2a-parameterize-taxyear`)
3. The primary file to understand is `tmd/datasets/cps.py` (799 lines)
4. Key companion files: `tmd/datasets/tmd.py`, `tmd/datasets/taxcalc_dataset.py`,
   `tmd/imputation_assumptions.py`, `tmd/datasets/puf.py`
5. PR #3 goal: add RawCPS_2022, CPS_2022, PUF_2022 classes and make tmd.py year-aware
6. Plan file: `~/.claude/plans/enumerated-gliding-willow.md`
7. Key findings: CPS has 175M tax units (137M filers, 38.5M nonfilers); .5806 scale reduces
   nonfilers to 22.3M; Treasury OTA benchmark is 32M nonfilers; PUF_2022 needed for uprating
