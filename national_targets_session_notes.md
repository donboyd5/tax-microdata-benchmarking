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

## Answers to Questions (updated 2026-03-01)

1. **Recipe file version**: `target_recipes_v4.xlsx` is current. Exists on both master and `improve-potential-targets-structure` branches. `setup.R` on master references v2; on `improve-potential-targets-structure` the QMD uses v4.

2. **potential_targets_preliminary.csv status**: **RESOLVED.** File exists on master (1.2 MB, 6,281 rows). Contains data for 2015, 2021, 2022. Has 41 unique var_names. Schema: rownum, idbase, year, table, var_name, var_type, var_description, value_filter, subgroup, marstat, incsort, incrange, ptarget, fname, xlcell, xl_colnumber, xlcolumn, xlrownum.

3. **New variables for 2022 vs 2021**: **[PENDING]**

4. **New targets to add**: **[PENDING]**

5. **Crosswalk output format**: Separate file (JSON or CSV). Don confirmed this.

6. **Long-term goal**: Convert the entire target-getting process from R to Python. Don confirmed: OK to do this early rather than finishing R pipeline first.

## Open Questions (updated 2026-03-01)

- ~~What's in `data/potential_targets_preliminary.csv`?~~ **RESOLVED** — 6,281 rows, 41 var_names, 3 years
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

## Session Update: 2026-03-01

### What we learned this session:
- Currently on `master` branch (Clarabel reweighting PR already merged)
- `improve-potential-targets-structure` branch exists locally (79 commits ahead of master, tracks origin)
- `potential_targets_preliminary.csv` exists on master with all 3 years of data (6,281 rows, 41 var_names)
- `target_recipes_v4.xlsx` exists on both branches
- `setup.R` on master references v2; on improve-potential-targets-structure branch the QMD uses v4
- `pufirs_fullmap.json` maps PUF→IRS (reverse of what we need for Step 2)
- `soi.csv` has 47 unique TMD variable names, 2 years (2015, 2021), 5,331 rows

### Agreed execution order:
1. **Steps 2-3 first: Create mapping files** (IRS var_name → PUF → TMD)
   - Extract from existing code: `clean_vname()` in soi_targets.py, `pufirs_fullmap.json`, crosswalk QMD docs
   - Don reviews and verifies
2. **Step 4: Write Python converter** (potential_targets_preliminary.csv + mappings → soi.csv)
3. **Step 5: Verify 2021** (compare new soi.csv to current, run reweighting)
4. **Step 6: Add 2022** (generate 2022 soi.csv, update reweighting)
5. **Step 1 (later): Port R→Python** (read IRS Excel files directly in Python)

### Plan file location:
`~/.claude/plans/shimmering-meandering-wand.md`

---

## Session Update: 2026-03-03

### What we accomplished:

#### Phase 1 Complete: Tier 1 Mapping Validation Framework
- Created `tmd/national_targets/data/irs_to_puf_map.json` — maps IRS var names to PUF codes
- Created `tmd/national_targets/data/puf_to_tmd_map.json` — maps PUF codes to TMD variable names
- Created `tmd/national_targets/validate_mappings.py` — three-level validation script
- Created `tmd/national_targets/test_validate_mappings.py` — automated regression test
- **10/10 Tier 1 variables validated successfully**

#### Three-Level Validation Chain
We discovered that the correct validation approach has three levels:

1. **Check 1 — IRS extraction (IRS spreadsheet vs soi2015 from PUF booklet p.46)**
   - Both are full SOI sample totals → should match exactly if mapping is correct
   - Result: 10/10 exact match (0.0000% difference)

2. **Check 2 — Documentation errors (soi2015 vs soi2015_adj)**
   - Flags 10x missing-digit errors in PUF documentation (Don identified these in R code)
   - 13 variables have this issue (E17500, E18400, E18500, E19200, E19700, etc.)
   - None of the 10 Tier 1 variables are affected

3. **Check 3 — PUF quality (soi2015_adj vs PUF calculated sum)**
   - Measures PUF subsample representativeness (expected ~0.5-2% diff)
   - NOT a mapping error — it's subsampling variation
   - Result: 10/10 within 5% threshold

#### Key technical lessons:
- **Must use "All returns, total" row (incsort==1)** from potential_targets to avoid double-counting (detail rows sum to same total)
- **Must use E-codes** (raw PUF inputs) not C-codes (computed) for validation against pufsums
- **Value filter matters**: AGI uses `value_filter='all'`, most income types use `'nz'` (nonzero)
- The original comparison (IRS vs PUF sum) conflated mapping correctness with PUF subsampling quality; the three-level chain separates these concerns

#### Tier 1 validated variables (IRS name → PUF code → TMD name):
| IRS var | PUF code | TMD variable | IRS vs SOI | SOI vs PUF |
|---------|----------|--------------|------------|------------|
| agi | E00100 | adjusted_gross_income | 0.0000% | +0.40% |
| wages | E00200 | employment_income | 0.0000% | +0.62% |
| taxint | E00300 | taxable_interest_income | 0.0000% | +2.05% |
| orddiv | E00600 | ordinary_dividends | 0.0000% | +0.26% |
| pensions | E01500 | total_pension_income | 0.0000% | +0.84% |
| socsectot | E02400 | total_social_security | 0.0000% | -0.06% |
| socsectaxable | E02500 | taxable_social_security | 0.0000% | -0.16% |
| ti | E04800 | taxable_income | 0.0000% | +0.60% |
| taxbc | E05800 | income_tax_before_credits | 0.0000% | +0.72% |
| taxac | E08800 | income_tax_after_credits | 0.0000% | +0.70% |

#### Phase 2: Gain/Loss Variables (Tier 2)

Extended validation to handle gain/loss variables (those with value_filter gt0/lt0).

**Gain/loss validation approach:**
- Check 1 becomes: IRS_gt0 - IRS_lt0 should equal soi2015 (net value confirms mapping)
- Check 3 splits: IRS_gt0 vs pufsums.sumgtz, IRS_lt0 vs |pufsums.sumltz|

**Results:**
| IRS var | PUF code | IRS net vs SOI | Status |
|---------|----------|---------------|--------|
| busprofincome | E00900 | 0.0000% | validated |
| cgtaxable | E01000 | 0.0000% | validated |
| partnerscorpincome | E26270 | 0.0000% | validated |
| rentroyalty | E25850 | 406.66% (mismatch!) | needs review |
| estateincome | E26390 | -19.11% (mismatch) | needs review |
| partnerincome | E26270 | N/A (2021/2022 only) | no 2015 data |
| scorpincome | E26270 | N/A (2021/2022 only) | no 2015 data |

**Key findings requiring detective work:**
- **rentroyalty**: E25850 is "Rent/royalty net income" but has a 10x documentation error.
  Even with adjustment, the IRS definition of rentroyalty doesn't match E25850.
  Multiple IRS columns (rent, royalty, farm rent, etc.) may need combining.
- **estateincome**: IRS gt0-lt0 ($27.4B) != E26390 soi2015 ($33.9B).
  Possible definitional mismatch between IRS table and PUF variable.
- **E02000** (Schedule E net) combines rent + royalties + partnerships + S corps + estate/trust.
  Individual IRS components (rentroyalty, partnerincome, scorpincome, estateincome) are
  subsets that don't map 1:1 to PUF codes.
- **partnerincome** and **scorpincome** are new 2021/2022 breakdowns of the combined
  partnerscorpincome (E26270). They can't be validated against 2015 pufsums.

### Commits made:
1. `8606cb2` Add Phase 1: IRS-PUF-TMD mapping validation framework
2. `212912b` Add automated test for Tier 1 mapping validation
3. `e05a649` Restructure validation with three-level comparison chain
4. `de58deb` Phase 2: gain/loss validation

#### Phase 3a: Quick-Win Variables (Tier 3a)

Added 5 straightforward variable mappings. All pass three-level validation:

| IRS var | PUF code | TMD variable | IRS vs SOI | SOI vs PUF |
|---------|----------|--------------|------------|------------|
| qualdiv | E00650 | qualified_dividends | 0.0000% | +0.41% |
| pensions_taxable | E01700 | taxable_pension_income | 0.0000% | +0.44% |
| unempcomp | E02300 | unemployment_compensation | 0.0000% | -2.06% |
| exemptint | E00400 | exempt_interest | 0.0000% | -0.39% |
| cgdist | E01100 | capital_gains_distributions | 0.0000% | +1.77% |

#### Detective Work: Schedule E Component Decomposition

**Major finding**: The three IRS Schedule E components sum to E02000 soi2015 **exactly**:

```
rentroyalty net:         $103,059M - $46,246M = $56,813M
partnerscorpincome net:  $755,623M - $126,618M = $629,005M
estateincome net:         $32,453M -  $5,033M =  $27,420M
─────────────────────────────────────────────────────────
SUM:                                            $713,238M
E02000 soi2015:                                 $713,238M  ← EXACT MATCH
```

**Why individual components don't match their PUF codes:**

1. **estateincome** — Symmetric excess ($1.44B on BOTH income and loss sides)
   - IRS gt0 ($32.5B) < E26390 soi2015 ($33.9B) — PUF includes extra items
   - IRS lt0 ($5.0B) < E26400 soi2015 ($6.5B) — same extra items on loss side
   - BUT the nets match exactly: $27.4B = $27.4B
   - Likely: PUF includes passive activity reclassifications symmetrically

2. **rentroyalty** — Asymmetric excess (PUF is larger on both sides, more so for losses)
   - IRS gt0 ($103.1B) < E25850_adj ($112.1B) — 8.8% difference
   - IRS lt0 ($46.2B) < E25860 ($59.8B) — 29.3% difference
   - Farm rental (E27200=$4.5B) is included in IRS rentroyalty but separated in PUF
   - PUF variables may capture pre-passive-limitation amounts
   - Nets don't match: IRS $56.8B vs PUF $52.4B — still needs manual review

3. **partnerscorpincome** — Perfect match at net level ($629.0B = $629.0B)

**Targeting implications**:
- For estateincome: IRS gt0/lt0 values could be used as targets even though they differ from PUF code soi2015, because the net is correct and the differences are definitional
- For rentroyalty: Still needs manual review to decide targeting strategy

### Commits made:
1. `8606cb2` Add Phase 1: IRS-PUF-TMD mapping validation framework
2. `212912b` Add automated test for Tier 1 mapping validation
3. `e05a649` Restructure validation with three-level comparison chain
4. `de58deb` Phase 2: gain/loss validation
5. `34cdf3c` Phase 3a: quick-win variables + Schedule E detective work

### Updated plan file:
`~/.claude/plans/bubbly-splashing-volcano.md`

---

## Validated Variables Summary (as of Phase 3a)

**Total: 22 variables mapped, 18 validated, 2 flagged for review, 2 no 2015 data**

| Tier | Variables | Status |
|------|-----------|--------|
| Tier 1 (standard) | agi, wages, taxint, orddiv, pensions, socsectot, socsectaxable, ti, taxbc, taxac | 10/10 validated |
| Tier 2 (gain/loss validated) | busprofincome, cgtaxable, partnerscorpincome | 3/3 validated |
| Tier 2 (needs review) | rentroyalty, estateincome | 2/2 flagged |
| Tier 2 (no 2015 data) | partnerincome, scorpincome | 2021/2022 only |
| Tier 3a (quick wins) | qualdiv, pensions_taxable, unempcomp, exemptint, cgdist | 5/5 validated |

**Remaining unmapped** (20 of 42 var_names):
iradist, exemption, exemptions_n, id, id_salt, id_mortgage, id_retax, id_pit, id_gst,
id_pitgst, id_taxpaid, id_intpaid, id_contributions, id_medical_capped, id_medical_uncapped,
itemded, sd, qbid, amt, tottax

---

## Session Update: 2026-03-04

### What we accomplished:

#### Phase 4 Complete: All 42 Variables Mapped, 100% soi.csv Coverage

Extended `irs_to_puf_map.json` from 22 to all 42 IRS var_names. Achieved 47/47 soi.csv variable coverage (100%).

**New variables mapped:**

| Tier | Variables | Status |
|------|-----------|--------|
| Tier 4 standard | iradist (E01400), tottax (E06500) | 2/2 validated |
| Tier 4 itemized deductions (10x errors) | id_medical_uncapped (E17500), id_contributions (E19700), id_intpaid (E19200), id_retax (E18500), id_salt (E18400) | 5/5 validated (with expected 10x doc errors) |
| Variables without PUF E-codes | exemption, exemptions_n (XTOT), amt (C09600), qbid (QBIDED), itemded (C21060), sd (C04200) | Mapped (TC computed or no E-code) |
| Deduction subset variables | id_mortgage, id_pit, id_gst, id_pitgst, id_taxpaid, id_medical_capped (C17000) | Mapped (subsets of E-codes, no independent validation) |
| Count-only variable | id (count of itemized filers) | Mapped as tmd_name=null (skipped in converter) |

**Key fixes:**
- Fixed 6 TMD name mismatches between our mapping chain and soi.csv conventions
- Added `tmd_name_count: "count"` to agi entry for return count mapping
- Updated validate_mappings.py with 10x error fallback logic
- Added Tier 4 tests (standard + 10x) to test_validate_mappings.py

#### Deduction Hierarchy Discovery

```
id_taxpaid (C18300) ⊃ id_salt (E18400) + id_retax (E18500) + personal property taxes
id_salt (E18400)    = id_pit + id_gst (filers choose one)
id_intpaid (E19200) ⊃ id_mortgage + investment interest
```

Variables like id_mortgage, id_pit, id_gst are subsets of their parent E-code — no separate PUF codes exist.

#### Python Converter: potential_targets → soi.csv

Created `tmd/national_targets/potential_targets_to_soi.py` — replaces the old `agi_targets.csv → soi_targets.py → soi.csv` pipeline.

**Key design:**
- Reads `potential_targets_preliminary.csv` + `irs_to_puf_map.json`
- Maps var_name+var_type+value_filter → TMD variable name using irs_to_puf_map.json
- Maps marstat → Filing status, incrange → AGI bounds, subgroup → Taxable only
- Handles partner+S-corp aggregation for 2021 (partnerincome+scorpincome → partnership_and_s_corp)
- ptarget is already in dollars (old pipeline had values in thousands)
- Does NOT produce Taxable only rows (potential_targets has no taxfilers subgroup; reweight.py skips these anyway)

**Converter output:** 3,850 rows (vs old 5,331 — difference is missing Taxable only rows)

#### End-to-End Verification: `make clean && make data`

Ran full pipeline with converter-generated soi.csv:

1. **558/558 targeted rows** match exactly between new and old soi.csv (zero mismatches)
2. **1,986/1,986 non-taxable 2021 rows** match exactly
3. **Reweighting targeted 558 SOI statistics** (same count as before)
4. **All 14 tests passed** (including test_income_tax, test_tax_revenue, test_variable_totals)
5. **Growth factors identical** (md5 checksum match)
6. tmd.csv.gz and weights checksums differ only due to gzip timestamps + GPU float non-determinism

Old soi.csv backed up at `tmd/storage/input/soi.csv.bak`.

### Commits made this session:
(See git log on `improve-potential-targets-structure` branch for full list)

---

## Validated Variables Summary (as of Phase 4)

**Total: 42 variables mapped, 25 validated against pufsums, 47/47 soi.csv variables covered**

| Tier | Variables | Status |
|------|-----------|--------|
| Tier 1 (standard) | agi, wages, taxint, orddiv, pensions, socsectot, socsectaxable, ti, taxbc, taxac | 10/10 validated |
| Tier 2 (gain/loss validated) | busprofincome, cgtaxable, partnerscorpincome | 3/3 validated |
| Tier 2 (needs review) | rentroyalty, estateincome | 2/2 flagged |
| Tier 2 (no 2015 data) | partnerincome, scorpincome | 2021/2022 only |
| Tier 3a (quick wins) | qualdiv, pensions_taxable, unempcomp, exemptint, cgdist | 5/5 validated |
| Tier 4 (standard) | iradist, tottax | 2/2 validated |
| Tier 4 (10x errors) | id_medical_uncapped, id_contributions, id_intpaid, id_retax, id_salt | 5/5 validated |
| Mapped (no pufsums) | exemption, exemptions_n, amt, qbid, itemded, sd, id_mortgage, id_pit, id_gst, id_pitgst, id_taxpaid, id_medical_capped, id | 13 mapped |

## Session Update: 2026-03-05

### What we accomplished:

#### Quality Control Analysis: Two Key Comparison Tables

Produced two QC comparison tables and saved them in `tmd/national_targets/data/qc_reports/`:

**1. `targets_year_comparison.csv` — IRS targets across 2015, 2021, 2022**
- 96 rows: all unique (var_name, var_type, value_filter) combinations at total-income-range level (incsort=1, marstat='all')
- Columns: var_name, var_description, var_type, value_filter, 2015/2021/2022 values, 2022-2021 difference, % change
- Key findings:
  - Wages up 7.9% ($9,022B → $9,739B) 2021→2022
  - Capital gains (gt0) down 38% ($2,049B → $1,270B) — 2021 was a boom year
  - Unemployment comp collapsed 85.5% ($209B → $30B) — end of COVID benefits
  - Taxable interest up 29% ($104B → $134B) — rising interest rates
  - 2015-only variables: exemption, exemptions_n, partnerscorpincome (pre-TCJA)
  - 2021+ only variables: qbid, partnerincome, scorpincome, id_pitgst

**2. `puf_vs_irs_2015.csv` — PUF weighted sums vs IRS published totals for 2015**
- 63 rows: all mapped variables with both a PUF code and pufsums data
- Columns: var_name, var_description, var_type, value_filter, puf_code, irs_2015, puf_2015, difference, pct_diff
- Sorted descending by absolute % difference
- Source: PUF sums from `pufsums.csv` (derived from `puf_2015.csv` via `summarize_puf_for_irs_comparison.qmd`; **includes the 4 aggregate records**)
- Key findings:
  - Most variables within 1% — excellent PUF representation
  - Large discrepancies concentrated in Schedule E components:
    - estateincome lt0: +71% count, +23% amount (PUF includes passive activity items)
    - rentroyalty lt0: +31% amount, +25% count (PUF captures pre-limitation amounts)
    - rentroyalty gt0: +9% amount (farm rental separated in PUF, combined in IRS)
    - partnerscorpincome lt0: -6%
  - Core variables (wages, agi, dividends, pensions, social security, taxes) all <1%

#### Understanding the 558 Target Count

Documented how 11 AGI-level + 13 aggregate variables produce 558 targets:
- `adjusted_gross_income` and `count`: 19 AGI bins × 5 filing statuses = 95 each (190 total)
- Other 9 AGI-level income vars: 19 bins × 1 ("All") × 2 (amount + count) = 38 each (342 total)
- 13 aggregate vars: 1 × 2 (amount + count) = 2 each (26 total)
- Total: 190 + 342 + 26 = **558** (for year 2021; 2015 has 505 due to fewer bins)

#### Branch Merged with Master

Merged master into `improve-potential-targets-structure` branch (clean merge, no conflicts). Branch now includes:
- Clarabel reweighting (#416), version 1.2.0 (#417), scipy deprecation fix (#418), AGI range label fix (#421)
- Full pipeline (`make clean && make data`) passes: 57 tests passed, 3 skipped, 0 failures
- Clarabel QP solver: 550 targets solved in ~10s, all within 0.5% tolerance

#### Correction: Target Count is 550, Not 558

The actual targeted count is **550**, not 558. The 8 difference:
- `estate_income`, `estate_losses`, `rent_and_royalty_net_income`, `rent_and_royalty_net_losses` are **commented out** in `reweight.py` (lines 90-96) because Tax-Calculator doesn't model them (all zeros in `tc_to_soi()`)
- Even if uncommented, `_drop_impossible_targets()` would remove them (all data values zero)

#### Deeper Schedule E Analysis

Investigated all PUF Schedule E variables to find combinations matching IRS totals:

**PUF Schedule E variables:**
- E25850: Rent/royalty income (positive values only, income-side)
- E25860: Rent/royalty loss (positive values representing losses, loss-side)
- E26270: Partnership/S-corp net income (true signed variable)
- E26390: Estate/trust income (positive values only, income-side)
- E26400: Estate/trust loss (positive values representing losses, loss-side)
- E27200: Farm rental income
- E02000: Schedule E net total = E25850 - E25860 + E26270 + E26390 - E26400 + E27200

**Best PUF combo for rentroyalty income**: E25850 - E27200 = $108.0B (+4.8% vs IRS $103.1B)
**Rentroyalty loss**: E25860 = $60.4B vs IRS $46.2B (+30.6%) — no good match

**Root cause**: PUF captures pre-passive-activity-limitation amounts; IRS reports post-limitation.
This causes systematic PUF > IRS for rental/estate losses.

**Decision**: Exclude rentroyalty and estateincome from 2022 targeting until better approach found.

#### Don's PUF-IRS Correspondence Notes (from earlier research)

Key findings from Don's detailed PUF-IRS correspondence work:
- **10x documentation errors confirmed**: E17500, E18400, E18500, E19200, E19700 have documentation values that are 10x too low (missing a digit)
- **Capital gains on aggregate records**: $86.7B (12.5% of total) is on the 4 aggregate PUF records (MARS=0)
- **SALT and aggregate records**: Question about whether aggregate records should be included in SALT comparisons
- **E02000 soi2015 matches IRS component net sum exactly**: $713.238B = rentroyalty net + partnerscorpincome net + estateincome net
- **pufsums.csv includes 4 aggregate records** (derived from `puf_2015.csv` via R Quarto script)

#### IRS Spreadsheet Provenance Added to Mapping

Updated `irs_to_puf_map.json` with `irs_locations` field for each variable, recording exact IRS Excel file and cell references (e.g., `"2015": "15in14ar.xls_G9"`). Sourced from `potential_targets_preliminary.csv` `fname` and `xlcell` columns.

#### 2022 Data Added to soi.csv

- Converter updated to support year-specific variable exclusions via `year_exclude_vars` parameter
- soi.csv now has 3 years: 2015 (1,864 rows) + 2021 (1,986 rows) + 2022 (1,830 rows) = **5,680 rows**
- 2022 excludes rentroyalty and estateincome (4 TMD variables: estate_income, estate_losses, rent_and_royalty_net_income, rent_and_royalty_net_losses)
- 2021 targeted rows verified: 558/558 match backup soi.csv exactly
- Full pipeline passes with new 3-year soi.csv (57 tests passed)

#### Targeted Variables Summary Document

Created `tmd/national_targets/data/targeted_variables_summary.md` documenting:
- All 20 targeted variables (11 AGI-level + 9 aggregate)
- How 550 targets are constructed (bins × filing statuses × amount/count)
- AGI income bins (19 bins)
- Variables NOT targeted (estate/rentroyalty — commented out in reweight.py)

#### Important: Pipeline Still Targets 2021

The current pipeline is hardcoded to `TAXYEAR = 2021` in `create_taxcalc_input_variables.py`. The 2022 targets are in soi.csv but to actually use them for reweighting would require:
- Changing `TAXYEAR` to 2022
- Having growth factors that age PUF from 2015 to 2022
- Ensuring Tax-Calculator models all targeted variables correctly for 2022

---

## Session Update: 2026-03-05 (continued — PR #424 merged)

### What we accomplished:

#### PR #424 Merged: 2022 IRS Target Data Infrastructure

Created and merged PR #424 (`pr-424-2022-target-infrastructure`) into PSLmodels master. This is the first of a planned 5-PR strategy for enabling 2022 national reweighting.

**Files merged:**
- `tmd/national_targets/potential_targets_to_soi.py` — Python converter (replaces old soi_targets.py pipeline)
- `tmd/national_targets/data/irs_to_puf_map.json` — 42-variable IRS→PUF→TMD mapping with irs_locations
- `tmd/storage/input/soi.csv` — 3-year data (2015/2021/2022, 5,680 rows)
- `tmd/national_targets/data/targeted_variables_summary.md` — documentation of 550 targets

**Verification:** `make clean && make data` passes — all tests identical to before merge. 2021 optimizer sees identical data.

**Lint:** Both `pycodestyle` and `pylint` pass with zero issues.

#### 5-PR Strategy Status

| PR | Description | Status |
|----|-------------|--------|
| PR #1 (PR #424) | Infrastructure (soi.csv, converter, mapping) | **MERGED** |
| PR #2a | Cleanup hardcoded 2021 (keep TAXYEAR=2021) | Pending |
| PR #2b | Flip TAXYEAR to 2022 (depends on #2a + #3) | Pending |
| PR #3 | CPS 2022 classes (parallel with #2a) | Pending |
| PR #4 | Test updates (depends on #2b) | Pending |
| PR #5 | All-Python, data-driven targets | Longer-term |

**Plan file:** `~/.claude/plans/sprightly-munching-finch.md` — contains detailed analysis of each PR.

#### Workflow Lessons Learned

- **Don't push to origin or create PRs without explicit user direction.** Don's workflow: Claude prepares branches and commits; Don pushes upstream and creates PRs.
- **Push to upstream (PSLmodels), not origin (donboyd5 fork)** when creating PRs for review.
- **Run `make format` and lint before committing** — black, pycodestyle, pylint.

---

## Session Update: 2026-03-05 (continued — PR #2a in progress)

### What we accomplished:

#### PR #2a Commit: Parameterize TAXYEAR

Branch: `pr2a-parameterize-taxyear` (1 commit: `b3a1b3f`)

**Core change:** Moved `TAXYEAR = 2021` to `imputation_assumptions.py` as the single source of truth. Eliminated all hardcoded `2021` references and duplicate `TAX_YEAR` aliases.

**Files modified (6):**

| File | Change |
|------|--------|
| `tmd/imputation_assumptions.py` | Added `TAXYEAR = 2021` |
| `tmd/create_taxcalc_input_variables.py` | Import TAXYEAR instead of defining it |
| `tmd/utils/reweight.py` | Removed `TAX_YEAR` alias, use `TAXYEAR` directly in function defaults |
| `tmd/utils/reweight_clarabel.py` | Import `TAXYEAR` from imputation_assumptions (was importing `TAX_YEAR` from reweight) |
| `tmd/create_taxcalc_cached_files.py` | Removed `TAX_YEAR` alias, use `TAXYEAR` directly |
| `tmd/datasets/tmd.py` | Replaced 5 hardcoded `2021` values with `TAXYEAR` |

**Why imputation_assumptions.py?** `tmd.py` and `create_taxcalc_input_variables.py` import each other → circular import if TAXYEAR lives in either. `imputation_assumptions.py` has zero tmd imports.

**Verification with TAXYEAR=2021:** 51 tests passed, 3 skipped (identical to master baseline).

**Testing with TAXYEAR=2022:** `make data` crashes at `create_taxcalc_cached_files.py` because `tc.Records.tmd_constructor()` hardcodes `start_year=2021` internally. The growth factors also have a subtlety at lines 51-57. These need fixing to make the pipeline 2022-ready.

#### Remaining work for PR #2a:

Two fixes needed so that `make data` runs (even if tests fail) with TAXYEAR=2022:

1. **`create_taxcalc_cached_files.py`**: Bypass `tmd_constructor()` and construct `tc.Records()` directly with `start_year=TAXYEAR`
2. **`create_taxcalc_growth_factors.py`**: Lines 51-57 use `gfdf.iat[2022 - FIRST_YEAR, ...]`. When FIRST_YEAR=2022, index=0 overwrites the baseline all-ones row. Must be conditional on FIRST_YEAR < 2022.

---

## Resume Instructions

When resuming this session:
1. Read `repo_conventions_session_notes.md` first
2. Currently on **`pr2a-parameterize-taxyear`** branch (1 commit: `b3a1b3f`).
3. **PR #424 merged.** Infrastructure for 2022 targets is in production master.
4. **IMMEDIATE TASK**: Fix the two remaining issues so `make data` completes with TAXYEAR=2022:
   - **`create_taxcalc_cached_files.py`**: `tc.Records.tmd_constructor()` hardcodes `start_year=2021`. Fix: construct `tc.Records()` directly with `start_year=TAXYEAR`.
   - **`create_taxcalc_growth_factors.py`**: Lines 51-57 growth factor adjustments must be conditional (`if FIRST_YEAR <= 2021`).
5. **After fix**: Test with TAXYEAR=2021 (all tests pass), test with TAXYEAR=2022 (`make data` completes, tests may fail on hardcoded fingerprints), commit, then Don pushes.
6. **SUBSEQUENT STEPS** (from 5-PR strategy):
   - **PR #2b**: Actually change TAXYEAR default to 2022. Depends on PR #3.
   - **PR #3**: CPS 2022 classes (RawCPS_2022, CPS_2022). CPS 2022 data URL already in cps.py.
   - **PR #4**: Update tests for year flexibility (hardcoded fingerprint values).
   - **PR #5**: All-Python pipeline, data-driven targets.
7. **Workflow reminders**:
   - Don't push or create PRs without explicit user direction
   - Run `make format` and lint (pycodestyle + pylint) before committing
   - Don pushes upstream and creates PRs; Claude prepares branches and commits
8. Key files:
   - `tmd/imputation_assumptions.py` — `TAXYEAR = 2021` — single source of truth (NEW location)
   - `tmd/create_taxcalc_growth_factors.py` — 2022 growth factor adjustments at lines 51-57
   - `tmd/create_taxcalc_cached_files.py` — needs `tmd_constructor` bypass
9. **Plan file:** `~/.claude/plans/sprightly-munching-finch.md` — detailed analysis of all PRs
10. This session notes file is at `session_notes/national_targets_session_notes.md`
