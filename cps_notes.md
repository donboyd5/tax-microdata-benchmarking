**Read `repo_conventions_session_notes.md` first.**

### **DO NOT PUSH TO UPSTREAM OR CREATE UPSTREAM PRs WITHOUT EXPLICIT USER PERMISSION**

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

**CORRECTED ANSWER**: All 550 SOI targets are **filer-only** targets. In `build_loss_matrix()`
(`reweight.py:124`), every target mask includes `* filer` where `filer = (DATA_SOURCE == 1)`,
i.e., PUF records only. CPS records (`DATA_SOURCE == 0`) contribute **zero** to every target.

The prescale at `reweight_clarabel.py:403` (`flat_file["s006"] *= prescale`) applies to ALL
records including CPS, which is a design smell — it adjusts nonfiler weights for no purpose.
Since CPS records have zero in every target column, Clarabel has no incentive to change their
weights (any change is pure deviation cost with no target benefit).

**Effective CPS weight flow:**
```
CPS nonfiler weights (from family weights):  38.5M
  × CPS_WEIGHTS_SCALE (0.5806):              22,328,784 (intended nonfiler count)
  × prescale (filer ratio, applied to ALL):  22,278,615 (unintentional distortion)
  → Clarabel optimizer:                       ~no further change (no targets apply)
```

**The prescale is a bug for CPS records.** It applies a filer-derived ratio
(`SOI_filer_target / current_PUF_filer_total`) to ALL records including CPS nonfilers
(`reweight_clarabel.py:403`). This overwrites the nonfiler count that .5806 was intended
to produce. In the current run the distortion is small (-50K, -0.2%) because PUF weights
happened to be close to the SOI target. But if PUF weights were further off, the distortion
would be proportionally larger.

**Bottom line:** The .5806 is the only intentional nonfiler weight control. There is no
nonfiler population target to validate it against. The prescale then unintentionally
changes the nonfiler count. Clarabel effectively ignores CPS records (no targets apply).
There are no total-population targets.

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

4. **PolicyEngine dependency**: Three packages provide the Dataset base class, the
   Microsimulation class, and variable metadata. PolicyEngine does NOT compute taxes —
   Tax-Calculator does that (`add_taxcalc_outputs` in tmd.py:40). PolicyEngine is used as a
   data access and aggregation layer: it reads variables from h5py files, aggregates
   person-level data to tax-unit level, and determines filer/nonfiler status (by comparing
   income to filing thresholds). The `pe_sim.calculate("variable_name")` calls in
   `create_tc_dataset()` are reading stored CPS/PUF data, not running tax simulations.

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

## Appendix: CPS Weighted Sums by Variable, 2021 vs 2022

Person-level weighted sums using `A_FNLWGT/100`. CPS 2021 = income year 2021 (March 2022 survey,
asecpub22csv.zip, 152,732 records, 327.8M weighted persons). CPS 2022 = income year 2022
(March 2023 survey, asecpub23csv.zip, 146,133 records, 329.7M weighted persons).

Sorted by absolute percent change, descending. Dollar amounts in billions (B) or millions (M).

### Tax Credits & Federal Tax (Census Tax Model)

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| CTC_CRD | Child tax credit | 4.4B | 91.1B | +86.7B | +1958.6% |
| ACTC_CRD | Additional child tax credit | 203.7B | 22.1B | -181.6B | -89.1% |
| FEDTAX_AC | Federal tax after credits | 1,098.5B | 1,649.1B | +550.6B | +50.1% |
| EIT_CRED | Earned income tax credit | 46.8B | 35.7B | -11.1B | -23.7% |
| TAX_INC | Taxable income | 9,922.3B | 10,392.9B | +470.6B | +4.7% |
| AGI | Adjusted gross income | 12,787.3B | 13,385.4B | +598.1B | +4.7% |
| MARG_TAX | Marginal tax rate | 2.1B | 2.2B | +84.9M | +4.0% |
| STATETAX_B | State tax (method B) | 440.6B | 447.5B | +7.0B | +1.6% |
| STATETAX_A | State tax (method A) | 413.0B | 406.9B | -6.2B | -1.5% |
| FEDTAX_BC | Federal tax before credits | 1,718.8B | 1,706.9B | -11.8B | -0.7% |

**Notes:** CTC_CRD jumps +1959% because 2021 had the expanded fully-refundable CTC ($3,600/$3,000
per child, delivered mainly through ACTC_CRD). In 2022, CTC reverted to $2,000/child (non-refundable
portion = CTC_CRD), so ACTC_CRD collapsed -89%. FEDTAX_AC rose +50% as refundable credits shrank.
UC_VAL fell -81% as pandemic unemployment benefits expired.

### Income Variables (Person-Level)

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| UC_VAL | Unemployment compensation | 88.7B | 16.5B | -72.2B | -81.4% |
| CAP_VAL | Capital gains/losses | 367.4B | 177.4B | -190.0B | -51.7% |
| INT_VAL | Interest income | 594.2B | 342.4B | -251.8B | -42.4% |
| FRSE_VAL | Farm self-employment income | 31.5B | 40.5B | +9.0B | +28.5% |
| PAW_VAL | Public assistance (TANF) | 5.8B | 7.4B | +1.6B | +27.9% |
| WC_VAL | Workers compensation | 9.7B | 7.8B | -1.9B | -19.8% |
| DIS_VAL1 | Disability income (source 1) | 32.3B | 38.5B | +6.2B | +19.1% |
| OI_VAL | Other income | 14.0B | 15.9B | +1.9B | +13.9% |
| DIV_VAL | Dividend income | 209.9B | 181.8B | -28.1B | -13.4% |
| WSAL_VAL | Wage and salary income | 9,873.7B | 10,856.9B | +983.2B | +10.0% |
| SS_VAL | Social Security income | 953.6B | 1,043.1B | +89.4B | +9.4% |
| RETCB_VAL | Retirement contributions | 394.0B | 426.7B | +32.7B | +8.3% |
| ANN_VAL | Annuity income | 59.1B | 54.4B | -4.8B | -8.1% |
| PNSN_VAL | Pension income | 435.4B | 463.0B | +27.6B | +6.3% |
| SSI_VAL | Supplemental Security Income | 52.7B | 50.2B | -2.5B | -4.7% |
| CSP_VAL | Child support received | 19.8B | 20.7B | +881.6M | +4.5% |
| VET_VAL | Veterans benefits | 91.9B | 95.9B | +4.0B | +4.3% |
| RNT_VAL | Rental income | 218.9B | 212.2B | -6.7B | -3.1% |
| SEMP_VAL | Self-employment income | 491.7B | 497.8B | +6.1B | +1.3% |
| DIS_VAL2 | Disability income (source 2) | 745.6M | 337.1M | -408.6M | -54.8% |

**Notes:** CAP_VAL (-52%) and INT_VAL (-42%) reflect the 2022 market downturn and possible
CPS reporting differences. DIV_VAL (-13%) follows the same pattern. WSAL_VAL (+10%) reflects
wage growth. SS_VAL (+9.4%) reflects the 5.9% COLA for 2022.

### Retirement Distributions

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| DST_VAL2 | Ret. distribution (source 2) | 11.2B | 16.7B | +5.6B | +50.0% |
| DST_VAL2_YNG | Ret. dist. young (source 2) | 609.3M | 368.3M | -241.0M | -39.5% |
| DST_VAL1_YNG | Ret. dist. young (source 1) | 39.2B | 40.4B | +1.1B | +2.9% |
| DST_VAL1 | Ret. distribution (source 1) | 210.4B | 207.2B | -3.2B | -1.5% |

### Expense Variables

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| CHSP_VAL | Child support paid | 19.3B | 20.1B | +814.9M | +4.2% |
| MOOP | Medical out-of-pocket expenses | 603.7B | 627.3B | +23.6B | +3.9% |
| PHIP_VAL | Health insurance premiums | 317.2B | 329.0B | +11.8B | +3.7% |

### Work Variables

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| WKSWORK | Weeks worked per year | 7.8B | 8.1B | +291.7M | +3.7% |
| HRSWK | Hours worked per week | 6.5B | 6.6B | +147.0M | +2.3% |

### Demographics & Weights

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| A_FNLWGT | Person weight (total pop.) | 327.8M | 329.7M | +1.9M | +0.6% |
| A_AGE | Age (person-weighted sum) | 12.9B | 13.0B | +121.3M | +0.9% |
| A_SEX | Sex code (person-weighted) | 494.3M | 496.9M | +2.5M | +0.5% |
| A_MARITL | Marital status code | 1,408.1M | 1,407.7M | -387.7K | -0.0% |
| PRDTRACE | Race code | 513.3M | 519.0M | +5.8M | +1.1% |
| PRDTHSP | Hispanic origin code | 156.8M | 165.0M | +8.2M | +5.2% |
| A_HSCOL | School enrollment | 42.4M | 43.6M | +1.1M | +2.7% |
| MRK | Marketplace health coverage | 643.0M | 646.4M | +3.3M | +0.5% |
| WICYN | WIC receipt | 162.3M | 162.5M | +246.7K | +0.2% |

### Disability Flags (person-weighted sums of codes)

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| PEDISPHY | Physical disability | 458.9M | 465.2M | +6.3M | +1.4% |
| PEDISREM | Cognitive disability | 465.5M | 471.7M | +6.3M | +1.3% |
| PEDISDRS | Self-care disability | 471.2M | 477.5M | +6.3M | +1.3% |
| PEDISOUT | Going-out disability | 465.4M | 471.6M | +6.2M | +1.3% |
| PEDISEYE | Vision disability | 471.9M | 478.0M | +6.1M | +1.3% |
| PEDISEAR | Hearing disability | 467.3M | 473.3M | +6.0M | +1.3% |

### SPM Unit Variables

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| SPM_ACTC | SPM: Additional CTC | 944.5B | 107.5B | -837.0B | -88.6% |
| SPM_FEDTAX | SPM: Federal tax | 2,415.6B | 4,440.0B | +2,024.4B | +83.8% |
| SPM_POOR | SPM: Poverty status | 24.9M | 39.8M | +14.9M | +60.1% |
| SPM_WICVAL | SPM: WIC value | 7.5B | 11.5B | +3.9B | +52.3% |
| SPM_BBSUBVAL | SPM: Broadband subsidy | 2.1B | 2.9B | +752.8M | +35.5% |
| SPM_EITC | SPM: EITC | 180.7B | 152.3B | -28.3B | -15.7% |
| SPM_WKXPNS | SPM: Work expenses | 870.5B | 735.7B | -134.8B | -15.5% |
| SPM_CAPHOUSESUB | SPM: Housing subsidy | 74.5B | 64.3B | -10.2B | -13.7% |
| SPM_POVTHRESHOLD | SPM: Poverty threshold | 9,024.1B | 9,920.2B | +896.0B | +9.9% |
| SPM_CHILDCAREXPNS | SPM: Childcare expenses | 246.6B | 270.9B | +24.3B | +9.8% |
| SPM_FICA | SPM: FICA | 2,196.9B | 2,399.5B | +202.7B | +9.2% |
| SPM_CAPWKCCXPNS | SPM: Work/childcare (capped) | 1,060.6B | 965.8B | -94.7B | -8.9% |
| SPM_SCHLUNCH | SPM: School lunch subsidy | 124.4B | 134.2B | +9.7B | +7.8% |
| SPM_TOTVAL | SPM: Total income value | 37,756.6B | 40,300.2B | +2,543.7B | +6.7% |
| SPM_MEDXPNS | SPM: Medical expenses | 1,889.9B | 1,975.4B | +85.5B | +4.5% |
| SPM_CHILDSUPPD | SPM: Child support paid | 45.6B | 47.3B | +1.6B | +3.6% |
| SPM_ENGVAL | SPM: Energy subsidy | 7.2B | 7.0B | -217.3M | -3.0% |
| SPM_RESOURCES | SPM: Total resources | 29,370.9B | 29,698.1B | +327.2B | +1.1% |
| SPM_SNAPSUB | SPM: SNAP benefits | 156.3B | 154.7B | -1.6B | -1.0% |
| SPM_FEDTAXBC | SPM: Fed tax before credits | 4,755.4B | 4,699.8B | -55.5B | -1.2% |
| SPM_STTAX | SPM: State tax | 1,149.3B | 1,148.7B | -624.3M | -0.1% |
| SPM_NUMKIDS | SPM: Number of kids | 325.5M | 321.4M | -4.1M | -1.3% |
| SPM_NUMADULTS | SPM: Number of adults | 725.6M | 730.9M | +5.3M | +0.7% |
| SPM_NUMPER | SPM: Number of persons | 1,051.1M | 1,052.3M | +1.2M | +0.1% |
| SPM_WEIGHT | SPM: Unit weight | 90,556.9B | 94,245.3B | +3,688.5B | +4.1% |

### Source Code Variables (weighted sums of codes — interpret with caution)

| Variable | Description | Wtd Sum 2021 | Wtd Sum 2022 | Change | % Change |
|----------|-------------|-------------:|-------------:|-------:|---------:|
| OI_OFF | Other income type code | 35.0M | 51.9M | +16.9M | +48.3% |
| DIS_SC2 | Disability source code 2 | 177.3K | 219.3K | +42.0K | +23.7% |
| DST_SC2 | Ret. dist. source code 2 | 2.6M | 3.2M | +603.8K | +23.3% |
| DST_SC2_YNG | Ret. dist. source 2 (young) | 189.8K | 154.4K | -35.4K | -18.6% |
| I_SEVAL | Self-emp imputation flag | 10.9M | 9.7M | -1.2M | -11.4% |
| DST_SC1_YNG | Ret. dist. source 1 (young) | 5.3M | 4.8M | -457.3K | -8.6% |
| DIS_SC1 | Disability source code 1 | 14.3M | 13.1M | -1.2M | -8.2% |
| DST_SC1 | Ret. dist. source code 1 | 30.0M | 31.8M | +1.8M | +6.0% |
| I_ERNVAL | Earnings imputation flag | 429.6M | 431.0M | +1.4M | +0.3% |

### Key Takeaways for CPS 2022 Support

1. **Largest changes are policy-driven, not data-format issues**: CTC/ACTC reversal (+1959%/-89%),
   unemployment expiration (-81%), and capital gains market effects (-52%) are real economic changes.
   The CPS variable *structure* is stable.

2. **No missing variables**: All 63 PERSON_COLUMNS, 10 TAX_UNIT_COLUMNS, and 39 SPM_UNIT_COLUMNS
   are present in both years. SPM_BBSUBVAL (broadband) is available in both 2021 and 2022.

3. **Population stable**: Total weighted persons grew only +0.6% (327.8M → 329.7M). Sample size
   fell from 152,732 to 146,133 records (-4.3%), but weights compensate.

4. **Income growth reasonable**: Wages +10%, Social Security +9.4% (5.9% COLA), pension +6.3%,
   retirement contributions +8.3%. These align with known economic conditions.

5. **No red flags for mechanical CPS 2022 support**: The data format is compatible. The large
   percentage changes are economically explained and don't indicate data problems.

---

## Implementation Results (PR #3)

### Files Modified (7 files)

| File | Change | Purpose |
|------|--------|---------|
| `tmd/datasets/cps.py` | +RawCPS_2022, +CPS_2022, +create_cps_2022(), parameterized retirement limits | Year-matched CPS dataset classes |
| `tmd/datasets/puf.py` | +PUF_2022, +create_pe_puf_2022() | Year-matched PUF class (time_period=2022 controls uprating target) |
| `tmd/datasets/tmd.py` | TAXYEAR-aware dataset selection, e01500/e00600 consistency clamps | Orchestration selects 2021 or 2022 classes based on TAXYEAR |
| `tmd/utils/pension_contributions.py` | TAXYEAR-aware CPS class selection | Pension imputation uses year-matched CPS |
| `tmd/utils/reweight_clarabel.py` | Prescale fix: only apply to filer records | Bug fix: CPS nonfilers were getting filer-derived prescale |
| `tests/test_weights.py` | Updated weight fingerprint | Reflects prescale fix (small diffs in total, mean, p50, p75) |
| `tests/test_imputed_variables.py` | Updated auto_loan_interest stats, ALL affpct | Reflects prescale fix ripple effect |

### Bug Fixes

1. **Prescale leak onto CPS nonfilers** (`reweight_clarabel.py:403`): `flat_file["s006"] *= prescale`
   applied the filer-count-derived prescale to ALL records. Fixed to
   `flat_file.loc[filer_mask, "s006"] *= prescale`. Effect on 2021 results: small — weight total
   shifted from 183.89M to 183.94M (+0.03%).

2. **e01500 >= e01700 constraint** (`tmd.py:60-61`): PUF uprating from 2015 to 2022 uses different
   growth factors for total vs taxable pension income, causing `total < taxable` for some records.
   Added `np.maximum` clamp after concat, before `add_taxcalc_outputs`. Same treatment for
   `e00600 >= e00650` (dividends).

### Test Results

- **TAXYEAR=2021**: 51 passed, 3 skipped (all green after updating fingerprints for prescale fix)
- **TAXYEAR=2022**: 45 passed, 3 skipped, 6 failed (all failures are hardcoded 2021 fingerprints)
  - `test_income_tax`: $187B actual vs $184B expected (+1.6%, plausible for 2022)
  - `test_weights`: different record count (226,366 vs 225,256) and weight distribution
  - `test_imputed_variable_distribution`: different imputation stats
  - `test_obbba_deduction_tax_benefits`: downstream of changed imputations
  - `test_tax_exp_diffs`, `test_tax_revenue`: baseline-year-dependent expected files
  - All 6 are "PR #4 scope" — updating expected values for TAXYEAR=2022

### Retirement Contribution Limits (parameterized)

| Year | 401(k) limit | 401(k) catch-up | IRA limit | IRA catch-up |
|------|-------------|-----------------|-----------|-------------|
| 2021 | $19,500 | $6,500 | $6,000 | $1,000 |
| 2022 | $20,500 | $6,500 | $6,000 | $1,000 |
| 2023 | $22,500 | $7,500 | $6,500 | $1,000 |

### Open Items for Later PRs

- Rename `create_tmd_2021()` to something year-neutral (cosmetic debt)
- Update 6 test fingerprints for TAXYEAR=2022 (PR #4)
- Consider year-specific `CPS_WEIGHTS_SCALE` values
- Consider year-specific imputation fractions

---

## Resume Instructions

When resuming this session:
1. Read `repo_conventions_session_notes.md` first
2. Currently on **`cps-exploration`** branch (based on `pr2a-parameterize-taxyear`)
3. PR #3 implementation is **complete** — code changes done, TAXYEAR=2021 tests all pass
4. Next step: PR to merge cps-exploration → master (or create PR #4 for 2022 fingerprints)
5. Plan file: `~/.claude/plans/enumerated-gliding-willow.md`
6. Key files changed: cps.py, puf.py, tmd.py, pension_contributions.py, reweight_clarabel.py,
   test_weights.py, test_imputed_variables.py
