**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Clarabel QP Reweighting PR

*Branch: `clarabel-reweighting` (to be created from upstream/master)*
*Last updated: 2026-02-27*

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

## Related Session Notes

- `nlp_reweighting_session_notes.md` — Clarabel/IPOPT experiments, cross-machine
  reproducibility confirmation, constraint scaling, UC tolerance experiments
- `reweighting_experimentation_session_notes.md` — scipy L-BFGS-B experiments,
  PyTorch cross-machine divergence root cause analysis
- `reweighting_session_notes.md` — PR #407 (PyTorch L-BFGS improvements)
