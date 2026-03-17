# State Weights Clarabel PR

## Session Notes — 2026-03-17

### Branch: `state-weights-clarabel`

### Commits (8)
1. `bb56d06` — Add Clarabel QP solver and share-based state targeting pipeline
2. `2333b5a` — Use separate target/weight subfolders for states and CDs
3. `2c87ea6` — Exclude PR and US from state target file generation
4. `c2b7afe` — Change default SOI area data year from 2021 to 2022
5. `7b65f57` — Add cross-state quality summary report
6. `bfd320a` — Print SOI year in prepare and solve stage messages
7. `a9da9a6` — Optimize quality report: load data once, read only needed columns
8. `ff9241d` — Switch to full 15-variable recipe, update variable mapping to shared names

### PR Strategy (revised 2026-03-17)
- **Single recipe with all 15 variable combos** (~146 targets per state), not a two-phase safe/extended approach
- **Two PRs to upstream** (mirroring national reweighting approach):
  - PR 1: Target preparation pipeline (prepare/ modules, recipe, mapping CSV)
  - PR 2: Solver + quality report (builds on PR 1's target files)
- **Two branches**: Branch 1 = target prep only, Branch 2 = solver on top
- User will run the actual push/PR commands

### Key Findings — 2026-03-17 Session
- Expanded from 91 "safe" targets to full 15-variable recipe (all from `ALL_SHARING_MAPPINGS`)
- 6 states (CA, DC, GA, MD, UT, VA) failed with `PrimalInfeasible` at slack_penalty=1e6
  - Root cause: numerical conditioning — absolute slack penalty creates scale mismatch when targets range from thousands to hundreds of billions
  - Lower penalty (1e4, 1e2, 10) progressively fixes more states
  - Relative slack scaling (penalty on % violation) fixed infeasibility but made violations too cheap — 6889 violations
- **e26270 (partnership/S-corp) is most problematic variable**: 12.8% sum-of-states vs national discrepancy, 131% mean baseline error, up to 11,500% for individual state/bin combos
- Stale `pr_targets.csv` and `us_targets.csv` found in targets dir — removed
- **Next steps**:
  - Exclude e26270 temporarily until data discrepancy is investigated
  - May need to regenerate TMD data after rebase (PRs #452, #453 changed master)
  - Add `agi_exclude` for low AGI bins on hard-to-hit variables
  - Add "largest violation" stat to solve summary
  - Re-run with clean data to match March 13 quality (51/51 solved, 85 violations)

### Sum-of-State Targets vs National Totals
| Variable | Diff% | Notes |
|----------|-------|-------|
| c00100 (AGI) | +0.37% | Good |
| e00200 (wages) | -1.02% | Good |
| e00300 (interest) | -4.08% | Slight concern |
| e01500 (pensions) | -1.12% | Good |
| e02400 (SS) | -4.83% | Some concern |
| e18400 (SALT inc) | -0.43% | Good |
| e18500 (SALT RE) | -0.43% | Good |
| **e26270 (partner/S)** | **+12.78%** | **Investigate** |

### x=1 Baseline Error by Variable (mean across states)
| Variable | Mean |abs err| | Max |abs err| |
|----------|----------------|----------------|
| e26270 | 131.6% | 11,502% |
| e18500 | 92.3% | 1,600% |
| e18400 | 86.5% | 3,079% |
| e00200 | 85.6% | 9,562% |
| c00100 | 70.1% | 7,248% |
| e00300 | 49.1% | 942% |
| e02400 | 26.1% | 240% |
| e01500 | 25.3% | 197% |

### Usage
```bash
# Full pipeline:
python -m tmd.areas.prepare_and_solve --scope states --workers 8

# Prepare targets only:
python -m tmd.areas.prepare_and_solve --scope states --stage targets

# Solve only (targets must exist):
python -m tmd.areas.prepare_and_solve --scope states --stage solve --workers 8

# Quality report:
python -m tmd.areas.quality_report
```
