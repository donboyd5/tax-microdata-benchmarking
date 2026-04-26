# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-26 (end of day — project closeout session)

## Active branch

`master` — unchanged from previous session, last commit `9ecd3fe`
(PR #513). Working tree clean.

Local branch list still pruned: only `master`, `county-data`, and
`session-notes` remain.

## Where things stand

**Area-weights project closed out.**

- 3-page project summary report drafted at
  `scratch/area_weights_project_report.md`. Pandoc-converted to
  `.docx` and `.html` siblings in `scratch/`. The `.docx` was sent to
  PSL.
- **Issue #381 (umbrella)** closed. The closeout draft used was
  `/tmp/issue_381_closeout.md` — eight-section accomplishment-to-PR
  map mirroring the report's structure, plus a "beyond scope"
  section, Martin Holmer's contributions grouped by theme, an
  "earlier infrastructure" section, and a six-item future-improvements
  list. Every PR-author attribution in the closeout was verified via
  `gh api .../pulls/<n>` before posting; the existing task-list
  comment got a one-paragraph note at the top pointing readers to the
  closeout below.

**Lesson recorded.** During drafting I misattributed the Clarabel
migration (#416) and PRs #362/#483 to Martin based on issue #381's
task list, when in fact those are Don's. `gh api` would have caught
this in seconds. `memory/feedback_verify_before_asserting.md` was
updated with a third caught-instance entry and a "How to apply" line
that explicitly requires `gh api repos/<org>/<repo>/pulls/<n> --jq
'.user.login'` for every PR cited in any deliverable, before writing
the name. Do not infer authorship from issue task lists, merge logs,
or branch names.

## Loose ends from this session

- `scratch/area_weights_project_report.{md,docx,html}` and
  `/tmp/issue_381_closeout.md` are on disk, untracked. Keep, archive,
  or delete at the user's discretion.
- The report doesn't include an author byline (user removed it during
  editing); add manually in the Google Doc if needed.

## Open work threads (unchanged from previous session)

### Future-year revenue comparator design (#502) — still awaiting Martin

No movement. Four-option menu sits in the issue body. No action
expected from us until Martin responds.

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

None.
