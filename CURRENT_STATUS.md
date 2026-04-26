# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-26 (later in day)

## Active branch

`master` — fast-forwarded to `9ecd3fe` after **PR #513** (issue #506
PR 2 of 2) merged. Working tree clean.

Local branch list pruned: only `master`, `county-data`, and
`session-notes` remain. Eleven merged feature branches were deleted
this session.

## Where things stand

**Issue #506 ("Improve area weights documentation") is DONE.** Both
PRs landed in the same session.

| Piece | PR | Branch | What it did |
|---|---|---|---|
| User docs | **#512** (merged) | `docs-areas-user` (deleted) | Restructured `tmd/areas/README.md` and rewrote `tmd/areas/weights/README.md` from scratch; added top-level README link; dropped Netlify references. Three commits including a year-mismatch fix to the pandas worked example. |
| Developer docs | **#513** (merged) | `docs-areas-developer` (deleted) | Audience preface + TOC for `AREA_WEIGHTING_GUIDE.md`; added `--congress` to all CD commands; repaired File Locations tree; replaced duplicate Quality Report section with pointer up to user README. Updated `AREA_WEIGHTING_LESSONS.md` "CDs not implemented yet" framing. Reconciled `prepare/recipes/README.md` with what the CLI actually loads (CSV specs current; `states.json` legacy/test-only; `cds.json` no longer exists). Added "Key entry points / used by" to `create_area_weights.py` docstring. |

PR 1's worked examples were verified end-to-end:
- pandas: `(tmd["e00200"] * wts["WT2022"]).sum()` = $1,371.9B for CA
  2022 wages (BEA-consistent scale).
- Tax-Calculator sketch: Records constructs cleanly; CA `iitax` 2024
  = $365B, year-aged wages = $1,538.9B.

PR 2 was a docs-only pass; no code changes other than the
`create_area_weights.py` docstring tweak.

## Open work threads

### Future-year revenue comparator design (#502) — still awaiting Martin

No movement since 2026-04-24. Four-option menu sits in the issue
body. No action expected from us until Martin responds.

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

None. The post-PR-513 cleanup deleted all 11 merged feature
branches in this session.
