# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-22 (session ~16:00 UTC)

## Active branch

`pr-498-review` — local tracking branch for PSLmodels PR #498 (martinholmer).
Fast-forwarded to PR head `29d576f` (2 commits: `soi_state_data.py` and
`soi_cd_data.py` PerformanceWarning refactors). Checked out so the user can
run `make clean && make data` to verify the PR.

## Open work threads

### 1. PR #498 review (issue #489) — user action pending
- User approved in a GitHub comment at 2026-04-22 14:57 UTC and suggested
  `df.assign(year=yr)` as an alternative phrasing.
- Martin then pushed a second commit extending the same pattern to
  `soi_cd_data.py`. Local branch now reflects both commits.
- **Next:** user runs `make clean && make data` on `pr-498-review`; if clean,
  no further action from us — PR is martin's to merge.

### 2. Pipeline warning hygiene (issue #430) — branch ready, not yet PR'd
- Branch: `pipeline-warning-hygiene`, 2 commits ahead of master.
  - `410038f` Capture pipeline warnings and surface test-run warnings
  - `c4b3eeb` Expand warning capture to all five pipeline stages (per-stage
    log files to avoid interleaving under `make -j`; `make warnings` scans
    all logs)
- **Open question:** PR it now, or first run `make data` end-to-end on this
  branch to verify warning capture actually works? Leaning toward verify-first.
- Related side branch: `delete-skipped-test-create-file` (also touches #430;
  deletes a skipped test). Decide whether to fold into the main PR, ship
  separately, or drop.

### 3. Conversation-resume hygiene (this thread)
- Built-in: `claude --resume` / `claude -c` restores full past sessions
  (stored in `~/.claude/projects/<project-hash>/*.jsonl`).
- This file is the complementary fast-recall layer.

## Pending/incoming from upstream

- **SALT growth rate PR** — coming from upstream. Rebase and rerun area
  weights when it merges. See `memory/project_salt_growth_improvement.md`.

## Branches to keep (do not prune)

`county-data`, `origin/least_squares_jvp`, `session-notes` — all have
unmerged work the user wants preserved. See MEMORY.md for details.
