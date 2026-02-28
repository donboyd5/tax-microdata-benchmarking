**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Clarabel QP Reweighting PR

*Branch: `clarabel-reweighting` (created from upstream/master)*
*Last updated: 2026-02-28*

---

## Goal

Create a PR that:
1. Replaces the default PyTorch L-BFGS reweighting with the Clarabel QP solver
2. Retains scipy L-BFGS-B as a backup (penalty-based approach)
3. Has quiet production output by default, verbose mode for debugging
4. Includes a weights test checking np.allclose() on the reproducibility fingerprint
5. Includes a GitHub issue explaining the change

---

## Planning Session (2026-02-27)

### Key decisions made

- **File naming**: `reweight_clarabel.py` for the new Clarabel solver (separate
  from `reweight.py` since they use fundamentally different formulations)
- **Backup solver**: scipy L-BFGS-B (best penalty-based solver, proper
  projected-gradient bounds, analytical gradient, no clamping plateau problem)
- **Other solver backends**: Stripped from the PR (IPOPT, OSQP, scipy trust-constr
  all removed — only Clarabel in the new file)
- **Verbose control**: Both env var (`VERBOSE_REWEIGHT=1`) and function parameter
  (`verbose=True`), parameter takes precedence
- **UC targets**: Included (not dropped). Default ±0.5% tolerance for all targets
- **Slack penalty**: Reset to 1e6 (from 1e7 on experiment-nlp)
- **Solver selection env vars**:
  - Default: Clarabel QP
  - `PYTORCH_REWEIGHT=1`: PyTorch L-BFGS (original from PR #407)
  - `SCIPY_REWEIGHT=1`: scipy L-BFGS-B backup
- **Dependency installation**: Add `clarabel` to `setup.py` `install_requires`.
  `make data` runs `make install` → `pip install -e .` which auto-installs it.
  Same protocol as torch, scipy, etc.
- **Branch**: New branch `clarabel-reweighting` based on updated upstream/master.
  Cannot merge experiment-nlp directly (too many incremental/experimental commits).
- **Documentation tone**: Explain things clearly for people unfamiliar with QP/
  optimization terminology. PR description and code docstrings should use plain
  English.

### Cross-machine reproducibility clarification

- **Clarabel**: Confirmed cross-machine reproducible (tested on two machines,
  identical results — see nlp_reweighting_session_notes.md)
- **scipy L-BFGS-B**: Likely NOT perfectly reproducible cross-machine (flat loss
  surface + BLAS differences), but expected to be closer than PyTorch L-BFGS
  (no clamping plateau problem, proper projected-gradient bounds). Cross-machine
  testing will be done as part of PR preparation to characterize actual divergence.
- **PyTorch L-BFGS**: NOT cross-machine reproducible (confirmed divergent — see
  reweighting_experimentation_session_notes.md). Root cause: torch.clamp()
  creates flat gradient regions where FP non-associativity causes records to
  end up at opposite bounds on different hardware.

### Pre-PR preparation: cross-machine reproducibility check

Run all three solvers on both machines and compare fingerprints:
1. Clarabel (default) — expect identical
2. scipy L-BFGS-B — expect close but not identical; document actual divergence
3. PyTorch L-BFGS — expect significant divergence (known)
Record all fingerprints in this file when available.

### Source material

- `tmd/utils/reweight_nlp.py` on `experiment-nlp` → refactor into `reweight_clarabel.py`
- `tmd/utils/reweight.py` on `experiment-grad-norm` (the `if use_scipy:` block)
  → extract as `reweight_lbfgsb()` function

### Plan file

Full implementation plan at:
`/home/donboyd5/.claude/plans/nested-exploring-pie.md`

---

## Implementation (2026-02-28)

### Commits

1. `082133e` Add Clarabel QP solver as default reweighting method
   - New: `tmd/utils/reweight_clarabel.py`, `tests/test_reweight_clarabel.py`
   - Modified: `tmd/datasets/tmd.py`, `tmd/utils/reweight.py`,
     `tmd/imputation_assumptions.py`, `setup.py`, `Makefile`

2. `d84471a` Update test expected values for Clarabel weights
   - Modified: `tests/test_weights.py`, `tests/expected_tax_expenditures`,
     `tests/test_imputed_variables.py`

### Cross-machine reproducibility results

Clarabel confirmed cross-machine deterministic:
- Machine 1 (WSL2 Linux): `make clean && make data` → 51 passed, 3 skipped
- Machine 2: `make clean && make data` → 51 passed, 3 skipped
- Weight fingerprint identical on both machines:
  n=225256, total=183887555.48, mean=816.349200, sdev=1053.418296,
  min=0.107690, p25=28.652287, p50=422.506425, p75=1320.009182,
  max=16527.649530, sum_sq=400080813544.38

### Solver output (production mode)

- Status: Solved, 19 iterations, ~10s
- All 550 SOI targets within +-0.5%
- No elastic slack needed (all constraints satisfied directly)

### Test value changes from master

Tax expenditures (7 of 9 changed, max change $8.1B):
paytax 1381.8→1382.9, iitax 2237.2→2245.3, ctc 129.4→129.5,
eitc 77.8→77.9, niit -43.6→-44.4, cgqd 174.4→174.6, qbid 52.6→52.7

OBBBA deductions (7 of 12 changed):
OTM: totben 23.95→23.83, affpct 8.90→8.87, affben 1401→1397
TIP: totben 6.93→6.85, affben 1380→1370
ALL: totben 54.86→54.77, affben 1018→1015

Weight test overhauled from 2-stat (mean/sdev with rtol=7e-5) to
10-stat fingerprint with np.allclose defaults (rtol=1e-5).

### Test tolerances

All weight-dependent tests now use np.allclose defaults or tighter,
except:
- test_tax_expenditures: rtol=5e-5, atol=0.1 (handles .1f rounding)
- test_area_weights: rtol=0.005 (area optimization is approximate,
  unchanged by this PR)
- test_imputed_variables OBBBA: atol varies (handles round() arithmetic)

---

## Related Session Notes

- `nlp_reweighting_session_notes.md` — Clarabel/IPOPT experiments, cross-machine
  reproducibility confirmation, constraint scaling, UC tolerance experiments
- `reweighting_experimentation_session_notes.md` — scipy L-BFGS-B experiments,
  PyTorch cross-machine divergence root cause analysis
- `reweighting_session_notes.md` — PR #407 (PyTorch L-BFGS improvements)
