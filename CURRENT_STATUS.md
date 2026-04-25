# Current Status

One-screen "where were we" snapshot. Claude updates this at the end of each
working session. If this looks stale, trust git + the topic-specific notes in
this folder over this file, and tell Claude to refresh it.

**Last updated:** 2026-04-24 (session ~22:00 UTC)

## Active branch

`soi-sanity-checks` — 1 commit ahead of upstream/master. Pushed and filed
as **PR #508** (merged). Working tree should be clean. Local `master` should
be fast-forwarded after #508's merge.

## Where things stand

The skipped-tests cleanup project (#430 → #501 umbrella) is **done**. Both
issues are now closed. Only one open thread remains from that work:

| Issue | State | Notes |
|---|---|---|
| **#502** — Future-year revenue comparator design discussion | **OPEN, waiting on Martin** | Four-option menu and gap analysis are in the issue body. No action expected from us until Martin responds. Could take a week or two. |
| #430 — Original triage of six skipped tests | closed | Mapped each item to its resolver in the closing comment |
| #501 — Umbrella plan for the cleanup | closed | All four pieces landed (or moved to #502 in case of PR 3) |

## What landed this project

| Piece | PR | Branch | What it did |
|---|---|---|---|
| Cleanup of `test_create_file` | #496 (merged) | `delete-skipped-test-create-file` | Removed item 4 of #430 |
| Pipeline warning capture | #497 (merged) | `pipeline-warning-hygiene` | Replacement coverage for #496's deletion plus broader warning surfacing |
| **PR 1** — National-file reproducibility fingerprint | #504 (merged, closes #503) | `unweighted-file-fingerprint` | Added `tests/test_tmd_file_fingerprint.py` + reference JSON |
| **PR 4 (first batch)** — Cleanup of items #504 superseded | #507 (merged) | `cleanup-skipped-tests` | Deleted `test_tmd_stats.py` + `tmd.stats-expect*` + orphan 2021 YAML files |
| **PR 2** — 2022 SOI sanity checks | **#508 (merged)** | `soi-sanity-checks` | Added `tests/test_soi_sanity_2022.py` (5 checks at 1% tolerance), retired `test_variable_totals` + `test_misc::test_income_tax` + `test_misc::test_partnership_s_corp_income` |
| Docs side-chore | #500 (merged) | `docs-soi-references` | Added `tmd/national_targets/docs/README.md` + gitignore for SOI PDFs |

## Open work threads

### Future-year revenue comparator design (#502) — awaiting Martin

Verified 2022 CBO-vs-TMD gaps (using `test_tax_revenue.py` formulas, not bare
`iitax`):
- `iitax + refund` (PUF, weighted) = $2,253.9 B vs CBO CY22 = $2,605.3 B → **−13.5%**
- `payrolltax` (all records) = $1,342.8 B vs CBO CY22 = $1,503.2 B → **−10.7%**

Gap is structural, not a TMD bug. Plausible decomposition documented in #502:
Form 1041 ($30–40 B), Form 1042 net of refunds ($50–100 B), cash-vs-accrual
timing in 2022 ($50–100 B), late assessments ($10–30 B), MTS residual
methodology (indeterminate). Unexplained residual ~$80–210 B.

Cross-check: TMD `iitax` matches SOI `tottax` to 0.35% (verified by
`test_soi_sanity_2022`) — the CBO-vs-TMD gap is primarily a CBO-vs-SOI
definitional gap, not a TMD modeling problem.

Four options in #502: (1) growth-rate comparison, (2) level comparison with
wide tolerance, (3) narrower CBO/JCT/Treasury publication if one exists,
(4) population-only future-year check. No action until Martin chooses.

### Loose ends not blocking issue closure

- `test_imputed_variable_distribution` — staying in place per Martin's
  preference (memory: `project_test_imputed_variable_distribution.md`).
- `tests/expected_itax_rev_2022_data.yaml` and `expected_ptax_rev_2022_data.yaml`
  are still pre-OBBBA CBO values; they will be refreshed once #502's
  comparator approach is settled.

## Pending/incoming from upstream

- **SALT growth rate PR** — coming from upstream. Rebase and rerun area
  weights when it merges. See `memory/project_salt_growth_improvement.md`.

## Branches to keep (do not prune)

`county-data`, `origin/least_squares_jvp`, `session-notes` — all have
unmerged work the user wants preserved. See MEMORY.md for details.

## Stale local branches (ok to delete when convenient)

After their PRs merged, the following branches can be deleted locally
whenever:
- `unweighted-file-fingerprint` (PR #504)
- `pipeline-warning-hygiene` (PR #497)
- `cleanup-skipped-tests` (PR #507)
- `docs-soi-references` (PR #500)
- `soi-sanity-checks` (PR #508)
- `pr-498-review` (PR #498 long since merged)

## Memory additions and updates this session

- `feedback_writing_no_jargon.md` — avoid programmer-specific shorthand
  (hot path, orthogonal, brittle); standard quantitative terms (rtol,
  tolerance, supersedes, deadwood) are fine.
- `feedback_verify_tax_concepts_from_source.md` — read SOI docs
  (`tmd/national_targets/docs/p1304_2022.pdf`) and TaxCalc output_vars.html
  before proposing a comparator; do not reason from recollection.
- `feedback_verify_before_asserting.md` — never claim file/field
  comparability without checking; applies even to my own prior-message
  statements.
- `feedback_virtual_pr_numbering.md` — write "PR 1" not "PR #1" for
  unfiled PRs in issue/PR drafts; the `#` auto-links to real (unrelated)
  low-numbered issues/PRs.
- `feedback_current_status_file.md` — maintain this CURRENT_STATUS.md file.
- `project_test_imputed_variable_distribution.md` — Martin cares about
  this test; do not propose retiring it.
- MEMORY.md: PR push workflow clarified — **PR branches push to `upstream`,
  not `origin`.**
