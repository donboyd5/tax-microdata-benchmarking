---
name: fast_difficulty_diagnostic
description: Gap-from-proportionate as fast alternative to LP feasibility check
type: project
date: 2026-03-26
---

# Fast Difficulty Diagnostic

## Problem

The LP feasibility check (`--lp-only`) takes 17 minutes for 436 CDs.
Too slow for iterative recipe development.

## Solution: gap-from-proportionate

For each area and target, compute:

    gap = |soi_share / pop_share - 1|

This measures how far the area's target deviates from what it would
get under simple population-proportional allocation.

**Speed: 0.7 seconds for all 436 CDs** (vs 17 min for LP).

## Results

Top 5 hardest CDs by mean |gap|:
- NY12 (Manhattan): 229%
- CA36: 179%
- CA16: 148%
- FL23: 136%
- FL19: 134%

Bottom 5 easiest:
- NC13: 18%
- NC04: 19%
- VA06: 21%
- NC14: 21%
- VA05: 23%

## Correlation with solver difficulty

Areas with high mean gap are more likely to:
- Need solver overrides (dropped targets, raised tolerance)
- Have more violated targets
- Take longer to solve

This metric could replace the LP check for fast screening during
recipe development.  Future PR could add `--difficulty-all` flag.
