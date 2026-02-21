**Read `repo_conventions_session_notes.md` first.**

# Session Notes: National Targets Pipeline Improvement
*Branch: `improve-potential-targets-structure`*
*Created: 2026-02-18*

---

## Goal Summary

Improve the pipeline that prepares national reweighting targets from IRS published data. The new pipeline should:

1. Read IRS Excel files for 2015, 2021, 2022 using a "recipes" specification
2. Assign short, meaningful variable names (var_name) to IRS measures
3. Tag each measure with metadata (var_type, value_filter, subgroup, marstat, etc.)
4. Record IRS spreadsheet provenance (table, row, column, year)
5. Link IRS var_names to PUF variable names and to TMD variable names
6. Write potential targets + linkages to disk
7. Write a Python replacement for `soi_targets.py` that reads the new output instead of `agi_targets.csv`
8. Test on 2021 (verify same results as current system)
9. Test on 2022 (new)

---

## Repository Structure

```
tax-microdata-benchmarking/
├── tmd/
│   ├── national_targets/          # R Quarto book (improve-potential-targets-structure)
│   │   ├── data/
│   │   │   ├── 2015/             # IRS Excel files: 15in11si.xls, 15in12ms.xls, 15in14ar.xls, 15in21id.xls
│   │   │   ├── 2021/             # IRS Excel files: 21in*.xls
│   │   │   ├── 2022/             # IRS Excel files: 22in*.xls
│   │   │   ├── target_recipes_v4.xlsx  # KEY: recipes for reading IRS files (active version)
│   │   │   ├── target_recipes_v3.xlsx  # (older version, setup.R still references v3 -- needs update)
│   │   │   ├── potential_targets_preliminary.csv  # Output of prepare_potential_targets.qmd
│   │   │   ├── pufirs_fullmap.json  # PUF TC var codes → IRS var names (e.g. e00200 → wages)
│   │   │   └── pufsums.csv / puf_codecounts.csv  # PUF documentation values
│   │   └── qmd/
│   │       ├── _quarto.yml        # Book config
│   │       ├── setup.R            # Sets GITROOT, DATADIR, targfn (currently="target_recipes_v3.xlsx")
│   │       ├── prepare_potential_targets.qmd  # Reads recipes+IRS files → potential_targets_preliminary.csv
│   │       ├── examine_potential_targets.qmd  # Validates targets interactively
│   │       ├── summarize_puf_for_irs_comparison.qmd  # Summarizes PUF vs IRS
│   │       ├── puf_irs_crosswalk_tmd2025.qmd  # Documents PUF-IRS crosswalk (table-format, partial)
│   │       ├── documentation_of_tmd2025_targeted_variables.qmd  # Documents existing mappings
│   │       ├── map_puf_variable_names_to_irs_spreadsheet_locations.qmd  # Maps PUF→IRS locations
│   │       └── mapping.qmd        # (stub)
│   ├── storage/input/
│   │   ├── agi_targets.csv        # CURRENT INPUT to soi_targets.py (old pipeline, manually created)
│   │   └── soi.csv                # CURRENT OUTPUT from soi_targets.py (read by reweight.py)
│   └── utils/
│       ├── soi_targets.py         # CURRENT: cleans agi_targets.csv → soi.csv
│       ├── reweight.py            # Uses soi.csv to build loss matrix for L-BFGS optimization
│       └── soi_replication.py     # tc_to_soi(): converts TC output variables to SOI names
```

---

## Current Pipeline (master / improve-reweighting)

```
IRS data (manually scraped)
    → agi_targets.csv
    → soi_targets.py (cleans, normalizes)
    → soi.csv (standardized targets, read by reweight.py)
    → reweight.py (build_loss_matrix → L-BFGS optimization)
    → updated s006 weights
```

### `agi_targets.csv` schema:
```
table, datatype, year, fname, xlcolumn, xlrownum, incsort, incrange, vname, ptarget, table_description
```
- `table`: tab11, tab12, tab14, tab21
- `datatype`: filers | taxable
- `vname`: agi, wages, nret_wages, orddiv, etc.
- `ptarget`: values in thousands (amounts) or raw count
- Years: 2015, 2021

### `soi.csv` schema (input to reweight.py):
```
Year, SOI table, XLSX column, XLSX row, Variable, Filing status,
AGI lower bound, AGI upper bound, Count, Taxable only, Full population, Value
```
- `Variable`: TMD-internal names (adjusted_gross_income, employment_income, etc.)
- `Value`: in dollars (amounts), or raw count
- `Count`: True/False (True = count of returns with nonzero value)
- `Taxable only`: True/False (True = taxable returns only; reweight.py EXCLUDES these)

---

## New Pipeline (improve-potential-targets-structure branch)

```
IRS Excel files (2015, 2021, 2022)
    + target_recipes_v4.xlsx (hand-created mapping workbook)
    → R Quarto: prepare_potential_targets.qmd
    → potential_targets_preliminary.csv/rds
    → [MISSING: crosswalk to TMD variable names]
    → [MISSING: Python script to produce soi.csv]
    → soi.csv (same format, read by reweight.py)
```

### `potential_targets_preliminary.csv` schema (output of R Quarto):
```
rownum, idbase, year, table, var_name, var_type, var_description,
value_filter, subgroup, marstat, incsort, incrange, ptarget,
fname, xlcell, xl_colnumber, xlcolumn, xlrownum
```
- `var_name`: IRS concept name (wages, orddiv, nret_wages, etc.)
- `var_type`: count | amount
- `value_filter`: nonzero, gt0, all, etc. (new concept, more granular than old)
- `subgroup`: filers (or others)
- `marstat`: all | single | mfjss | mfs | hoh
- `ptarget`: amounts in dollars (NOT thousands); counts are raw
- `year`: 2015, 2021, 2022 (3 years now vs 2 before)

---

## Key Variable Name Layers

| Layer | Example | Location |
|---|---|---|
| PUF/TC variable | e00200 | pufirs_fullmap.json keys |
| IRS short name (vname) | wages | agi_targets.csv, potential_targets_preliminary.csv |
| TMD internal variable | employment_income | soi.csv, reweight.py lists |
| TC computed variable | E00200 (uppercase) | tc_to_soi() in soi_replication.py |

### The `soi_targets.py::clean_vname()` mapping (IRS name → TMD name):
```
agi → adjusted_gross_income
wages → employment_income
orddiv → ordinary_dividends
qualdiv → qualified_dividends
cggross → capital_gains_gross
cgloss → capital_gains_losses
cgdist → capital_gains_distributions
busprofnetinc → business_net_profits
busprofnetloss → business_net_losses
partnerscorpinc → partnership_and_s_corp_income
partnerscorploss → partnership_and_s_corp_losses
pensions → total_pension_income
pensionstaxable → taxable_pension_income
socsectot → total_social_security
socsectaxable → taxable_social_security
taxint → taxable_interest_income
iradist → ira_distributions
exemptint → exempt_interest
rentroyinc → rent_and_royalty_net_income
rentroyloss → rent_and_royalty_net_losses
unempcomp → unemployment_compensation
estateinc → estate_income
estateloss → estate_losses
ti → taxable_income
taxac → income_tax_after_credits
taxbc → income_tax_before_credits
(empty / nret_all) → count
```

### `tc_to_soi()` mapping (TC computed variable → TMD name):
```
C00100 → adjusted_gross_income
E00200 → employment_income
E00600 → ordinary_dividends
E00650 → qualified_dividends
C01000 (pos) → capital_gains_gross
C01000 (neg) → capital_gains_losses
E01100 → capital_gains_distributions
E00900 (pos) → business_net_profits
E00900 (neg) → business_net_losses
E26270 (pos) → partnership_and_s_corp_income
E26270 (neg) → partnership_and_s_corp_losses
E01500 → total_pension_income
E01700 → taxable_pension_income
E02400 → total_social_security
C02500 → taxable_social_security
E00300 → taxable_interest_income
E01400 → ira_distributions
E00400 → exempt_interest
E02300 → unemployment_compensation
QBIDED → qualified_business_income_deduction
C04800 → taxable_income
C09200-REFUND → total_income_tax
C05800 → income_tax_before_credits
IITAX → income_tax_after_credits
```

---

## reweight.py (improve-reweighting): Key Behavior

The `reweight()` function reads `soi.csv` and builds two lists of targeted variables:

### AGI-level targeted variables (targeted by AGI bin):
- adjusted_gross_income, count, employment_income, business_net_profits,
  capital_gains_gross, ordinary_dividends, partnership_and_s_corp_income,
  qualified_dividends, taxable_interest_income, total_pension_income, total_social_security

### Aggregate-level targeted variables (full population only):
- business_net_losses, capital_gains_distributions, capital_gains_losses,
  estate_income, estate_losses, exempt_interest, ira_distributions,
  partnership_and_s_corp_losses, rent_and_royalty_net_income, rent_and_royalty_net_losses,
  taxable_pension_income, taxable_social_security, unemployment_compensation

**Filtering rules:**
- Excludes rows where `Taxable only == True`
- AGI-level vars: requires non-infinite AGI bounds (i.e., specific bins)
- Aggregate vars: requires (-inf, +inf) bounds AND Filing status handling for marstat variants
- Filters out "impossible" targets (where all data values = 0)

**Optimizer:** L-BFGS with strong Wolfe line search, up to 200 steps
**Precision:** float64
**Device:** GPU (CUDA) if available, else CPU

---

## Work Remaining (in improve-potential-targets-structure)

### R Quarto side:
1. [ ] Fix setup.R to reference `target_recipes_v4.xlsx` (or confirm v3 vs v4)
2. [ ] Verify `prepare_potential_targets.qmd` runs cleanly and produces complete output for 2015, 2021, 2022
3. [ ] Complete `puf_irs_crosswalk_tmd2025.qmd` - document all var_name → PUF → TMD name mappings
4. [ ] Write/complete a final QMD chapter that writes the final crosswalk to disk as CSV or JSON

### Python side:
5. [ ] Write new Python script (replacement for `soi_targets.py`) that:
   - Reads `potential_targets_preliminary.csv`
   - Applies crosswalk: var_name → TMD variable name
   - Maps var_type/value_filter/marstat/subgroup → Count, Taxable only, Filing status, Full population
   - Produces `soi.csv` in the exact same format as today
6. [ ] Ensure `reweight.py` uses the new soi.csv without modification (it already reads the file path correctly)

### Testing:
7. [ ] Test on 2021: compare resulting soi.csv to current soi.csv (should match)
8. [ ] Test reweighting on 2021: verify near-identical optimization results
9. [ ] Test on 2022: run full pipeline, review results

---

## Answers to Questions (updated 2026-02-18)

1. **Recipe file version**: `target_recipes_v4.xlsx` is current. `setup.R` needs to be updated from v3 → v4.

2. **potential_targets_preliminary.csv status**: Data directory is gitignored. Don will share the data contents. **[PENDING - Don returning shortly]**

3. **New variables for 2022 vs 2021**: **[PENDING]**

4. **New targets to add**: **[PENDING]**

5. **Crosswalk output format**: Separate file (JSON or CSV). Don confirmed this.

6. **Long-term goal**: Convert the entire target-getting process from R to Python. The R Quarto approach is current but Python is the target. Plan: (a) finish the R pipeline first to validate the approach, (b) then rewrite in Python using `openpyxl`/`xlrd` for Excel reading.

## Open Questions (still pending)

- What's in `data/potential_targets_preliminary.csv`? (Don will share)
- Any structural differences in 2022 IRS spreadsheets vs 2021?
- What new targets to add for 2022 version (beyond current 11+13 variable lists)?
- Does `target_recipes_v4.xlsx` have a `puf2015_wtdsums` sheet (used by `summarize_puf_for_irs_comparison.qmd`)?
- How does `value_filter` in new pipeline map to `Taxable only` in soi.csv?

---

## Documents Possibly Needed

- **PUF documentation** (the booklet referenced in summarize_puf_for_irs_comparison.qmd): `data/2015 Public Use Booklet.pdf` is in the repo
- **target_recipes_v4.xlsx**: The current active recipe file - need its contents to understand var_name definitions
- Any notes or documentation you've written outside the repo about intended new targets

---

## Long-term Python Conversion Goal

Don wants the entire target-getting process converted from R to Python eventually.

### Current R pipeline (to be replicated in Python):
- Reads `target_recipes_v4.xlsx` (irs_downloads sheet + tab*_map sheets)
- Downloads/reads IRS Excel files for each year
- Extracts columns per mapping, pivots to long format
- Outputs `potential_targets_preliminary.csv`

### Python reading approach:
- Use `xlrd` for old `.xls` files (IRS files are xls format)
- Use `openpyxl` for `.xlsx` files (recipes workbook)
- Replicate `get_rowmap()`, `get_colmap()`, `get_irs_table()` R functions in Python

### Two-phase plan:
1. **Near term**: Finish/validate R pipeline → write Python crosswalk script (R produces potential_targets_preliminary.csv, Python converts it to soi.csv)
2. **Long term**: Replace R pipeline entirely with Python — read IRS Excel files directly in Python using the same recipes approach

---

## Resume Instructions

When resuming this session:
1. The key branch is `improve-potential-targets-structure`
2. National targets R Quarto project is at `tmd/national_targets/qmd/`
3. Data is at `tmd/national_targets/data/` (partially gitignored — Don will share)
4. Python optimization code is at `tmd/utils/reweight.py` (reads `tmd/storage/input/soi.csv`)
5. The goal is to produce a new `soi.csv` (same format as current) from the new pipeline's outputs
6. **MOST IMPORTANT NEXT STEP**: Get `potential_targets_preliminary.csv` data from Don, then:
   a. Fix `setup.R` (v3 → v4)
   b. Write var_name → TMD variable name crosswalk as JSON
   c. Write Python script to produce `soi.csv` from potential_targets_preliminary.csv + crosswalk
   d. Test 2021 match, then test 2022 new
7. This session notes file is at `session_notes/national_targets_session_notes.md`
8. See also: `tmd/national_targets/qmd/documentation_of_tmd2025_targeted_variables.qmd` for existing crosswalk table
