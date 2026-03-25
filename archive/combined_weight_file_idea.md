---
name: combined_weight_file
description: Future feature — single combined weight file for cross-area analysis
type: project
date: 2026-03-25
---

# Combined Weight File

## Motivation

When analyzing Tax-Calculator results across many areas (436 CDs, 3,000
counties), users want to compare easily — e.g., effective tax rates by CD,
EITC per capita across counties. Currently each area has a separate
`{area}_tmd_weights.csv.gz`, requiring a loop to load and combine.

## Proposed design

- Long format: `area, RECID, WT2022, WT2023, ...`
- Parquet output for compression and fast column/row filtering
- `--combined` flag on `solve_weights` writes `{scope}_weights.parquet`
  alongside per-area files
- Per-area files remain for backward compatibility

## Scale

- CDs: 215K records × 436 areas = 94M rows (~200 MB Parquet)
- Counties: 215K records × 3,143 areas = 676M rows (~1.5 GB Parquet)
- Wide format (215K × 3,143 columns) is possible but awkward

## Status

Idea noted 2026-03-25. Not yet implemented. Consider for a future PR
after county pipeline is working.
