**Read `repo_conventions_session_notes.md` first.**

# PR Strategy for CD Pipeline
*Created: 2026-03-23*

## Guiding Principle

Split the work into two PRs:
1. **Infrastructure PR** — changes that improve the existing state pipeline
   and are independently useful. Easier to review, lower risk.
2. **CD Pipeline PR** — the actual CD targeting and weighting, building
   on the infrastructure.

## PR 1: Infrastructure Improvements

**Branch:** `infrastructure-improvements` (off master)

Changes that benefit states AND set up patterns for CDs/counties:

### Quality report enhancements
- Auto-save to file (`--output` flag)
- Scope/timestamp header, cumulative + wall-clock solve time
- Aggregate multiplier histogram
- Weight distribution by AGI stub (national vs sum-of-areas)
- Per-bin bystander analysis (T/. markers for targeted vs dropped)
- Top-N per-area detail for large scopes
- Explanatory labels for each section

### Solver infrastructure
- `solver_overrides.py` — per-area override YAML read/write
- `create_area_weights.py` — LP feasibility pre-check,
  unreachable target detection, elastic slack with per-constraint
  penalties, multiplier_max passthrough
- `batch_weights.py` — override support, multiplier_max passthrough

### Target sharing improvements
- `target_sharing.py` — synthetic variable support
  (capgains_net, ctc_total) in compute_tmd_national_sums

### What to EXCLUDE from this PR
- CD-specific files (soi_cd_data.py, CD recipes, CD targets)
- CD-specific scope handling in solve_weights.py
- developer_mode.py (not yet needed for states)
- prepare_shares.py (new pipeline, not yet used by states)
- Target spec CSVs (new format, not yet used by states)

### Files changed (estimated)
- `quality_report.py` — enhanced
- `create_area_weights.py` — LP feasibility, slack penalties
- `batch_weights.py` — override support
- `solver_overrides.py` — new file
- `target_sharing.py` — synthetic vars
- `constants.py` — minor additions (AT_LARGE_STATES, helpers)
- `target_file_writer.py` — include_totals support
- `AREA_WEIGHTING_GUIDE.md` — new file

## PR 2: CD Pipeline

**Branch:** `cd-pipeline` (rebased on infrastructure PR)

All CD-specific functionality:

### CD data ingestion
- `soi_cd_data.py` — CD SOI data, 117th→118th crosswalk, population
- `prepare/data/soi_cds/22incd.csv` — CD SOI source data

### CD targeting
- `prepare_shares.py` — pre-computed shares (new pipeline)
- `prepare/recipes/cd_target_spec.csv` — flat CSV spec
- `prepare/data/cds_shares.csv` — pre-computed CD shares
- `prepare_targets.py` — spec-based target preparation

### CD solving
- `solve_weights.py` — --scope cds support
- `developer_mode.py` — auto-relaxation cascade
- `prepare/recipes/cd_solver_overrides.yaml` — per-area overrides

### CD quality
- `quality_report.py` — --scope cds support (already in infra PR)

### Files changed (estimated)
- New: `soi_cd_data.py`, `prepare_shares.py`, `developer_mode.py`,
  CD recipes/specs/shares/overrides
- Modified: `prepare_targets.py`, `solve_weights.py`, `constants.py`

## Sequence

1. Create `infrastructure-improvements` branch off master
2. Cherry-pick / rewrite infrastructure changes
3. PR to upstream, get merged
4. Rebase `cd-pipeline` on new master
5. PR the CD-specific changes

## What NOT to include in either PR
- State target spec redesign (keep old JSON recipe for states for now)
- County pipeline
- Optimization A+B (separate stashed branch)
- explore_cd_data.py (exploration script, not needed in repo)
- Session notes (stay on session-notes branch)

## Data file decisions
- Include: `22incd.csv` (CD SOI source, ~4800 lines)
- Include: `cd_target_spec.csv` (92 rows), `cd_solver_overrides.yaml`
- Discuss: `cds_shares.csv` (127K rows, ~8MB) — generated artifact.
  Could be .gitignored and regenerated, or committed for reproducibility.
- Exclude: weight files, target files, quality reports (all generated)
- Exclude: state_target_spec.csv (not used by state pipeline yet)
