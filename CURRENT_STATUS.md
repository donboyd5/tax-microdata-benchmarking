# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-22 (session ~21:00 UTC)

## Active branch

`unweighted-file-fingerprint` — off master, **no commits yet**. Created to
comply with "never work on master" rule. Will hold PR #1 code when we reach
that step. Local `master` is current with `upstream/master` at `cfe2f82b`.

## Open work threads

### 1. Skipped-tests plan (follow-on to #430) — drafting, awaiting Martin's input

**Four-piece plan** (drafted, not yet posted as GitHub issues):

- **PR #1** — Unweighted national-file fingerprint (parallel to
  `tests/test_fingerprint.py`). Per-column stats:
  `count / sum / weighted_sum / std / min / max` with `rtol=1e-3`. The
  `weighted_sum` stat is included specifically to lock weighted 2022 totals
  (marginal-only fingerprint can't guarantee that).
- **PR #2** — 2022 sanity checks against SOI only (~4 assertions). Current
  TMD vs SOI agreement is excellent (all within 3%). See table below.
- **Issue #3** — Future-year revenue sanity check. **Unresolved design
  question (see below).**
- **PR #4** — Deadwood cleanup after #1 and #2 merge.

**Draft issue files in `/tmp/`:**
- `/tmp/issue_skipped_tests_plan.md` — the umbrella issue tagging
  @martinholmer for review. User has been iterating on this; latest version
  includes the SOI-only PR #2 table with actuals and the CBO/TMD
  definition-alignment problem asking for Martin's advice.
- `/tmp/issue_unweighted_fingerprint.md` — PR #1 sub-issue draft.

**KEY UNRESOLVED ISSUE — CBO vs TMD definition alignment (Issue #3):**
TaxCalc `iitax` and `payrolltax` are NOT apples-to-apples with CBO's
"Individual Income Taxes" and "Payroll Taxes" budget lines.
- CBO itax = Treasury cash receipts net of refunds (includes 1041
  fiduciary, 1042 NRA withholding, timing effects). TaxCalc `iitax` is
  accrued 1040 liability only (`c09200 - refund`).
- CBO ptax = OASDI + HI + UI + federal retirement + Railroad (both shares,
  incl. SECA). TaxCalc `payrolltax` = FICA + SECA only.
- Observed 2022 gaps: itax **−17.6%**, ptax **−10.7%** (TMD vs CBO CY22
  interpolated). Structural, not bugs.
- Four proposed approaches in draft issue: (1) growth-rate normalization,
  (2) wide-tolerance levels, (3) find a narrower CBO/JCT/Treasury
  publication, (4) population-only future-year check. Asking Martin.

**Verified 2022 baseline facts (do not re-derive):**
- TMD `sum(XTOT * s006)` = 333.98M vs Census 333.996M (−0.005%)
- TMD `sum(s006)` PUF-only = 162.13M vs SOI 161.34M (+0.5%)
- TMD `sum(e00200 * s006)` PUF = $9.712T vs SOI $9.739T (−0.3%)
- TMD `sum(c00100 * s006)` PUF = $14.852T vs SOI $14.834T (+0.1%)
- TMD `sum(iitax * s006)` = $2,147.4B vs SOI after-credits $2,098.9B (+2.3%)
- CBO CY22 iitax (FY22 + 0.25·(FY23−FY22)) = $2,605.3B → −17.6% vs TMD
- CBO CY22 ptax = $1,503.2B → −10.7% vs TMD

**Verified CBO baseline status:** 2026-02-01 Winter baseline IS post-OBBBA
and usable for future-year targets (itax FY26 = $2,751.291B, FY33 =
$3,743.854B — both ~3% lower than pre-OBBBA shipped values, consistent
with OBBBA extending TCJA provisions). Source:
https://github.com/US-CBO/eval-projections/blob/main/input_data/baselines.csv

**In-repo sources for targets:**
- SOI 2022 aggregates: `tmd/storage/input/soi.csv` (1,831 rows for 2022)
- CBO population: `tmd/storage/input/cbo26_population.yaml` (2010–2075)
- CBO revenue FY: `tests/expected_{itax,ptax}_rev_2022_data.yaml`
  (**currently pre-OBBBA**; needs refresh from 2026-02-01 baseline before
  use in Issue #3)

**Ordering and next steps:**
1. User finishing edits to `/tmp/issue_skipped_tests_plan.md` — when done,
   user copy-pastes into a new upstream issue (PAT workaround; I can't
   create).
2. User copy-pastes `/tmp/issue_unweighted_fingerprint.md` as PR #1
   sub-issue.
3. Wait for Martin on the umbrella issue — especially the CBO/TMD
   alignment question for Issue #3.
4. Based on Martin's input, implement PR #1 and PR #2 independently; then
   PR #4 after both merge.

### 2. PR #498 review (issue #489) — user action pending
User approved; Martin pushed second commit; user was to run
`make clean && make data` on `pr-498-review` branch. Status from 2026-04-22
earlier session unchanged.

### 3. Pipeline warning hygiene — PR #497 open
Branch `pipeline-warning-hygiene` pushed; PR #497 open on upstream.
Referenced in skipped-tests umbrella issue as Related.

### 4. Conversation-resume hygiene (this thread)
Built-in: `claude --resume` / `claude -c` restores full past sessions
(stored in `~/.claude/projects/<project-hash>/*.jsonl`). This file is the
complementary fast-recall layer.

## Pending/incoming from upstream

- **SALT growth rate PR** — coming from upstream. Rebase and rerun area
  weights when it merges. See `memory/project_salt_growth_improvement.md`.

## Branches to keep (do not prune)

`county-data`, `origin/least_squares_jvp`, `session-notes` — all have
unmerged work the user wants preserved. See MEMORY.md for details.

## Lessons captured this session (now in memory)

- `feedback_verify_before_asserting.md` — new feedback memory. Never claim
  file/field/definition comparability without checking. Caught twice in
  this session: once for "no in-repo target" on a file I had already
  quoted from, once for assuming TMD `iitax` ≈ CBO "Individual Income
  Taxes" without checking definitions.
