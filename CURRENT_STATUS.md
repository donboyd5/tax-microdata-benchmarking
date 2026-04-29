# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-29 (end of day — area-pipeline deadwood cleanup)

## Active branch

`cleanup-area-deadwood` — local-only feature branch, ready to push to
upstream. Two commits ahead of `master`. Working tree clean. **Do
NOT push to origin** (per Git Push Policy in MEMORY.md — PR branches
go straight to `upstream`).

`master` is current with `upstream/master @ 5dbe40f` (PR #516
merged: removed obsolete `tests/test_tax_revenue.py`).

## Where things stand

**This session covered three things:**

1. **State-targets table correction.** The morning's project-summary
   report (`scratch/area_weights_project_report.md` and
   `.docx`/`.html` siblings) had Table 2 listing 91 state targets and
   omitting SALT. Correct count is **175 per state**, including SALT
   subcomponents. Cause: the table was based on the legacy 91-row
   `tmd/areas/targets/ca_targets.csv` instead of the production
   175-row `tmd/areas/targets/states/ca_targets.csv` that
   `create_area_weights.py:92` (`STATE_TARGET_DIR`) actually reads.
   Verified by reading the production CSV row by row, and confirmed
   all 51 state files have the same 175-row structure. The CD table
   (107/CD) was also verified and is correct as-written.

   Corrected HTML table written to `/tmp/state_targets_table.html`
   for paste into the Google Doc. The on-disk
   `scratch/area_weights_project_report.{md,docx,html}` files have
   NOT been updated yet — user did not ask.

2. **PR #516 (delete obsolete `test_tax_revenue.py`).** The
   indefinitely-skipped fiscal-year-cash-receipts test was
   superseded by PR #515 (`test_revenue_levels_cbo.py`,
   calendar-year liability comparator). PR #516 removed the dead
   test plus its two unused YAML data files. Merged.

3. **Deadwood cleanup branch — IN PROGRESS, ready for review.**
   See "Tomorrow's pickup" below.

## Tomorrow's pickup — `cleanup-area-deadwood` branch

Two commits, total 20 files changed, ~4276 lines removed, ~23 added:

- **`1157a70`** — Remove unused `tmd/areas/targets/prepare/` legacy
  directory (15 files). The old Quarto-doc location;
  `tmd/areas/prepare/` is the live one. Includes 7 SOI state CSVs
  (years 2015-2021) that were never read by current code (loader
  uses `tmd/areas/prepare/data/soi_states/` which only has 2022),
  4 CD doc files (`21/22 incddocguide`, extracted xlsx,
  `congressional2021.zip`), the redirect README, and 2 placeholder
  `.gitignore`s. Also trims `SOI_STATE_CSV_PATTERNS` and
  `SOI_CD_CSV_PATTERNS` in `tmd/areas/prepare/constants.py` to the
  one year actually used (2022).

- **`150e83b`** — Remove unused `tmd/areas/make_all.py` "CI driver"
  (not actually invoked by any Makefile/CI). Its only externally
  used helper (`time_of_newest_other_dependency`) was moved into
  `batch_weights.py` as `_time_of_newest_other_dependency`. Updates
  `create_area_weights.py` docstring + `tmd/areas/README.md` +
  `AREA_WEIGHTING_GUIDE.md` to drop `make_all` mentions.

`make format && make lint` clean. All 338 tests still collect.
Smoke-tested the moved helper.

**Open decision before pushing — the 4 CD doc files in commit 1.**
They were SOI-published CD documentation
(`21incddocguide.docx`, `22incddocguide.docx`,
`cd_documentation_extracted_from_21incddocguide.docx.xlsx`,
`congressional2021.zip`). No equivalents currently in
`tmd/areas/prepare/data/soi_cds/` (only `22incd.csv` lives there).
Commit 1 deletes them on the assumption that they can be
re-downloaded from SOI if needed. If you'd rather keep them in the
repo, restore from upstream and migrate before pushing:
```
git checkout master -- tmd/areas/targets/prepare/prepare_cds/data/data_raw/
git mv tmd/areas/targets/prepare/prepare_cds/data/data_raw/*.{docx,xlsx,zip} \
       tmd/areas/prepare/data/soi_cds/
git commit --amend --no-edit
```

**Steps to ship the PR:**
1. (Optional) Decide on the 4 CD doc files (keep-by-migration vs delete).
2. `make data` — required before merging any PR.
3. `git push -u upstream cleanup-area-deadwood`.
4. Open upstream PR (suggested title: "Remove area-pipeline
   deadwood: tmd/areas/targets/prepare/ and make_all.py").

## Local-only deadwood (no PR — Don can `rm` any time)

51 loose `tmd/areas/targets/*_targets.csv` files at the top of
`targets/` (NOT under `targets/states/`) — gitignored, never
committed. The legacy 91-target files (the source of this morning's
table-mistake). Active code reads `targets/states/` instead.

```
rm tmd/areas/targets/*_targets.csv
# but KEEP tmd/areas/targets/xx_targets.csv and xx_params.yaml —
# those are test fixtures used by tests/test_solve_weights.py
```

So: `rm` everything matching `*_targets.csv` at that level except
`xx_targets.csv`. `xx_params.yaml` is the only matching `.yaml` and
should be left alone.

## Loose ends from prior sessions (unchanged)

- `scratch/area_weights_project_report.{md,docx,html}` and
  `/tmp/issue_381_closeout.md` are on disk, untracked. The
  state-targets table inside the report is wrong (91/state); see
  item 1 above. Decide whether to update the on-disk copies or
  only the Google Doc.

## Open work threads

### SALT growth rate PR — still incoming from upstream

When it merges, rebase any in-progress branches and rerun area
weights. See `memory/project_salt_growth_improvement.md`.

### Session-notes worktree — `session-notes` branch

Still tracked on `origin/session-notes`. This file gets pushed each
time it's updated.

## Branches to keep (do not prune)

`county-data`, `session-notes` (worktree), and the
`origin/least_squares_jvp` branch on the fork — all have unmerged
work the user wants preserved. See MEMORY.md.

## Stale local branches

None now. `cleanup-area-deadwood` is the active feature branch
(intentional).
