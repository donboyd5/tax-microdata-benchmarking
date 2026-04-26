# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-26

## Active branch

`master` — fast-forwarded to `8475745` after **PR #512** (issue #506
PR 1 of 2) merged. Working tree clean.

About to start a new branch for **issue #506 PR 2 of 2** (developer
docs) — see "Open work threads" below for the plan.

## Where things stand

Issue #506 ("Improve area weights documentation") is being addressed
in two PRs.

### Issue #506 PR 1 of 2 — user docs (DONE)

Merged as **PR #512** (`docs-areas-user` branch, three commits:
`54e57b7`, `4e1dc78`, `4c5298e`). What landed:

- Restructured `tmd/areas/README.md` as the user front door:
  Quickstart via `make`, Outputs, single-area runs, "When to rerun
  which stage" decision table, Quality reports section split into
  multi-area + individual-area + single-area diagnostics. Documents
  `quality_report_per_area.csv` (the all-areas summary CSV).
- Rewrote `tmd/areas/weights/README.md` from scratch (was 2 lines,
  now ~200): weight-file naming, columns, solver-log contents,
  worked examples both with pandas alone and inside Tax-Calculator.
- Added "Sub-national area weights" section to top-level `README.md`
  pointing into `tmd/areas/`.
- Made `tmd/README.md`'s "areas" bullet a link.
- Removed references to retired Quarto/Netlify documentation sites
  (`tmd-areas-prepare-{state,cd}-targets.netlify.app`); replaced
  with in-repo cross-links.

Worked examples in `tmd/areas/weights/README.md` were verified
end-to-end:
- pandas: `(tmd["e00200"] * wts["WT2022"]).sum()` = $1,371.9B for CA
  2022 wages (consistent with BEA scale).
- Tax-Calculator sketch: Records constructs cleanly; computed
  CA `iitax` 2024 = $365B, year-aged wages = $1,538.9B.

Decisions baked into the docs:
- Two-PR split (user docs first, developer docs later).
- Netlify sites: dropped references entirely (option G3 — the user
  will review the sites manually before deletion and flag anything
  worth preserving when we get to PR 2).
- Tax-Calculator is presented as one option, not the prescribed
  one. Computing weighted area totals never requires it; weights
  enter only at aggregation time.

## Open work threads

### Issue #506 PR 2 of 2 — developer docs (NEXT)

Branch name: **`docs-areas-developer`** (from current master).
Lighter-effort scope than PR 1 per the user's guidance.

Plan items from the original two-PR proposal:

- **F.** Add an "Audience" preface and TOC to
  `AREA_WEIGHTING_GUIDE.md`; nudge user-overview content to the
  front, keep recipe-extension/developer mode in clearly-labelled
  sections. Decide whether to keep as one doc or split into
  `user_overview.md` + `developer_guide.md` — user said "split
  unless you think one doc is better"; re-evaluate at the start
  of PR 2.
- **G.** `AREA_WEIGHTING_LESSONS.md`: update stale "CDs not
  implemented yet" content inline (developer info; user-confirmed
  inline update). Leave the substantive lessons intact.
- **H.** `prepare/recipes/README.md`: clarify which spec format is
  current vs legacy (the GUIDE flags `cds.json` / `states.json` as
  "legacy" but the recipes README still describes `states.json` as
  the recipe).
- **I.** `targets/prepare/README.md`: already cleaned up in PR 1
  (Netlify references removed). No additional work expected unless
  the user salvages content from the Netlify sites before deleting
  them — flag this at PR 2 kickoff.
- **J.** Module-level docstrings on `prepare_targets.py`,
  `solve_weights.py`, `quality_report.py`, `developer_tools.py`,
  `create_area_weights.py`.

User confirmed the developer reader is also a tax-policy analyst
but with more programming knowledge. Standard quantitative terms
(rtol, tolerance, supersedes, deadwood) are fine; programmer
shorthand (hot path, brittle, orthogonal) is not.

### Future-year revenue comparator design (#502) — still awaiting Martin

No movement since 2026-04-24. Still a four-option menu in the
issue body. No action expected from us until Martin responds.

### SALT growth rate PR — still incoming from upstream

When it merges, rebase any in-progress branches and rerun area
weights. See `memory/project_salt_growth_improvement.md`.

## Branches to keep (do not prune)

`county-data`, `origin/least_squares_jvp`, `session-notes` — all
have unmerged work the user wants preserved. See MEMORY.md.

## Stale local branches (ok to delete when convenient)

After their PRs merged, the following branches can be deleted
locally whenever:
- `docs-areas-user` (PR #512, **just merged**)
- `unweighted-file-fingerprint` (PR #504)
- `pipeline-warning-hygiene` (PR #497)
- `cleanup-skipped-tests` (PR #507)
- `docs-soi-references` (PR #500)
- `soi-sanity-checks` (PR #508)
- `pr-498-review` (PR #498 long since merged)
