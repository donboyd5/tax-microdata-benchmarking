# Adding a New Year of IRS Data

This document describes how to extend the national targets pipeline to a new
tax year (e.g., 2023) when IRS SOI data become available.  It covers the
discovery process — figuring out which Excel column holds each variable in the
new year's files — and how to encode that into the pipeline.

## Overview of the pipeline

```
IRS Excel files (.xls)
    ↓  extract_irs_to_csv.py        [one-time per year]
data/extracted/{year}/{table}.csv
    ↓  build_targets.py
data/irs_aggregate_values.csv       [all years, all tables, not deduplicated]
    ↓  potential_targets_to_soi.py
tmd/storage/input/soi.csv           [deduplicated, ready for reweight.py]
```

The single source of truth for IRS file layouts is
`tmd/national_targets/config/table_layouts.py`.  Adding a new year means
updating that file and re-running the pipeline.

---

## Step 1 — Download the IRS Excel files

The four tables needed are from IRS Statistics of Income, Individual Returns:

| Table | File pattern | Content |
|-------|-------------|---------|
| tab11 | `{yy}in11si.xls` | All returns — AGI by income size |
| tab12 | `{yy}in12ms.xls` | All returns — Marital status |
| tab14 | `{yy}in14ar.xls` | All returns — Sources of income |
| tab21 | `{yy}in21id.xls` | Returns with itemized deductions |

Place the files in `tmd/national_targets/data/{year}/`.

---

## Step 2 — Discovery: find the column letters

IRS regularly adds, removes, or reorders columns between years.  You cannot
assume the same column letter as the prior year without verifying.

**The recommended approach is to work interactively with Claude.**

Start a session with something like:

> "I need to map column letters for the 2023 IRS SOI tables.  The files are
> in `tmd/national_targets/data/2023/`.  Please help me verify each variable
> by reading the Excel headers and spot-checking values against published
> IRS totals."

Claude can:
- Read the `.xls` files with `xlrd` and print the hierarchical column headers
- Cross-check candidate columns against IRS published totals (usually in the
  accompanying PDF documentation, e.g., `p1304(2023).pdf`)
- Compare with the prior year's layout to flag shifts

### What to watch for

**Common IRS changes between years:**
- New columns inserted (e.g., IRS added tips and overtime in 2022 for tab14,
  shifting all subsequent columns)
- New variables added (e.g., `qbid` was added in 2021 post-TCJA; `id_pitgst`
  added in 2021 for tab21)
- Variables removed or merged (e.g., `partnerscorpincome` was split into
  `partnerincome` + `scorpincome` in 2021)
- Data row range changes (first/last row shifts if IRS adds or removes
  income brackets)

**Verification steps for each variable:**
1. Print the IRS column header text (the `irs_col_header` field in the
   extracted CSV) — it should match the variable description
2. Check the "All returns" row (incsort=1) against the published total in
   the IRS PDF or online table
3. For variables with income/loss splits (gt0/lt0), verify that income +
   loss totals are plausible

### Spot-check script

After a trial extraction, run:

```python
import pandas as pd
df = pd.read_csv("tmd/national_targets/data/extracted/2023/tab14.csv")

# Print all column headers for inspection
for _, row in df[df.incsort == 1].iterrows():
    print(f"{row.xlcolumn:4s}  {row.var_name:25s}  {row.var_type:8s}  "
          f"{row.raw_value:20,.0f}  {row.irs_col_header}")
```

Compare the printed values against the IRS PDF table for the "All returns"
row.

---

## Step 3 — Update `table_layouts.py`

In `tmd/national_targets/config/table_layouts.py`:

1. Add the new year to `YEARS`:
   ```python
   YEARS = (2015, 2021, 2022, 2023)
   ```

2. Add entries to `FILE_NAMES`:
   ```python
   ("tab11", 2023): "23in11si.xls",
   # etc.
   ```

3. Add entries to `DATA_ROWS` (check first/last data rows in the spreadsheet):
   ```python
   ("tab11", 2023): (10, 29),
   # etc.
   ```

4. For each column spec in the four `TAB*_COLUMNS` lists, add the new year's
   column letter to the `cols` dict:
   ```python
   {
       "var_name": "wages",
       ...
       "cols": {2015: "G", 2021: "G", 2022: "G", 2023: "G"},  # add 2023
   }
   ```
   If a variable does not exist in the new year, omit it from `cols`.
   If a new variable appears in the new year, add a new spec entry.

5. Add a comment explaining any shifts:
   ```python
   # 2023: IRS inserted new column X before wages, shifting subsequent cols
   ```

---

## Step 4 — Run the pipeline

```bash
# Extract the new year (existing years are skipped unless --overwrite)
python tmd/national_targets/extract_irs_to_csv.py --years 2023

# Rebuild irs_aggregate_values.csv (includes all years)
python tmd/national_targets/build_targets.py

# Rebuild soi.csv
python tmd/national_targets/potential_targets_to_soi.py
```

---

## Step 5 — Verify

Run the test suite:

```bash
python -m pytest tests/test_national_targets.py -v
```

The tests check row counts, known aggregate totals, and structural integrity.
For a new year you will need to add spot-check entries to
`TestExtractedCSVs::test_all_returns_spot_check` with known IRS published
totals (wages, AGI, return count from the IRS PDF).

Also update the row count assertions in `TestPotentialTargets` and
`TestSoi` to reflect the new year's data.

---

## Prior discovery work (2015, 2021, 2022)

The column mappings in `table_layouts.py` for 2015, 2021, and 2022 were
verified through an extensive interactive session:

- Every column letter was cross-checked against the IRS column header text
  (captured in the `irs_col_header` field of the extracted CSVs)
- "All returns" totals were verified against IRS published figures for key
  variables (wages, AGI, return count)
- The marital status decomposition was verified: sum of
  single + mfjss + mfs + hoh return counts equals the "All" total
- Cross-table consistency was checked: where the same variable appears in
  multiple tables (e.g., agi count in tab11/tab12/tab14), values were
  compared and documented
- Year-specific structural differences were documented in comments:
  TCJA effects (exemptions removed 2018+, qbid added), IRS column
  insertions (tips/overtime in 2022 tab14), and the
  partnerscorpincome split (combined in 2015, split in 2021+)

The `irs_col_header` column in the extracted CSVs is the primary
audit trail: it records the exact IRS header text above each column,
making it possible to independently verify any column mapping by
inspection.
