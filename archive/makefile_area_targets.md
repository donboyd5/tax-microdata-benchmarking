# Makefile Targets for Area Weighting (discussion notes)

## The problem

`make data` builds TMD and runs national tests. Area weighting (shares,
targets, weights) is a separate pipeline that users run manually. This
causes confusion:
- CD tests skip because data doesn't exist yet
- Reviewers don't know what commands to run
- No single command to reproduce area results

## Makefile basics

A Makefile defines **targets** (things to build) with **prerequisites**
(what must exist first) and **recipes** (commands to run):

```makefile
target: prerequisite1 prerequisite2
	command1
	command2
```

`make target` runs the commands if prerequisites are newer than the
target (or if the target doesn't exist). `.PHONY` targets always run.

## Proposed design

```makefile
# Number of parallel workers (override with: make states WORKERS=16)
WORKERS ?= 8

# State area weighting pipeline
.PHONY=state-shares state-targets state-weights states

state-shares: tmd_files
	python -m tmd.areas.prepare_shares --scope states

state-targets: state-shares
	python -m tmd.areas.prepare_targets --scope states

state-weights: state-targets
	python -m tmd.areas.solve_weights --scope states --workers $(WORKERS)

states: state-weights
	pytest tests/test_prepare_targets.py -v -k "not CD"
	pytest tests/test_state_weight_results.py -v

# CD area weighting pipeline
.PHONY=cd-shares cd-targets cd-weights cds

cd-shares: tmd_files
	python -m tmd.areas.prepare_shares --scope cds

cd-targets: cd-shares
	python -m tmd.areas.prepare_targets --scope cds

cd-weights: cd-targets
	python -m tmd.areas.solve_weights --scope cds --workers $(WORKERS)

cds: cd-weights
	pytest tests/test_prepare_targets.py -v -k CD
```

## How it works

- `make states` — runs the full state pipeline (shares → targets → weights → tests)
- `make state-targets` — runs just shares + targets (no solving)
- `make cd-shares` — runs just CD shares
- `make cds WORKERS=16` — full CD pipeline with 16 workers

Each step depends on the previous, so `make states` runs them in order.
If shares already exist and are newer than tmd_files, `make state-targets`
skips the shares step. (Though since these are .PHONY, they always re-run.
For true dependency tracking we'd need file-based targets, which is more
complex.)

## Where to put it

Options:
A. Top-level Makefile — most discoverable, ~20 lines added
B. tmd/areas/Makefile — keeps area stuff separate, invoke with
   `make -C tmd/areas states`
C. Top-level Makefile with thin delegation to tmd/areas/Makefile

Recommendation: A (top-level). It's only ~20 lines and users expect
`make <target>` at repo root.

## Which PR?

Could go in PR 2 (where state pipeline changes) or PR 4 (where CD
pipeline is complete). PR 4 is cleanest since both state and CD
targets would be available.
