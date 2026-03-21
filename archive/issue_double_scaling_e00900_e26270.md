# Draft Issue: E00900 and E26270 Are Double-Scaled During PUF Uprating

## Summary

E00900 (Sch C business net profit/loss) and E26270 (partnership/S-corp net income/loss) are erroneously uprated twice in `uprate_puf()`, making their pre-reweighting magnitudes approximately 35.5% too large in 2022. These are substantial variables — E00900 positive income was $544B in 2022 SOI data and E26270 was $1.3T. The reweighter likely compensates at the aggregate level, but the overstated record-level amounts may distort the weight distribution.

## Background: How `uprate_puf()` works

`uprate_puf()` in `tmd/datasets/uprate_puf.py` scales 2015 PUF dollar amounts to the target year (2022) using SOI-based growth factors. It uses `get_growth()`, which computes a per-return growth factor from SOI aggregates:

```
per_ret_growth = (soi_target_year_amount / soi_2015_amount) / (soi_target_year_returns / soi_2015_returns)
```

This per-return factor is applied to each record's dollar amounts. Weights (s006) are scaled separately by the returns ratio.

The function applies growth in three sequential blocks:

- **Block 1 (lines 140-153)**: "Straight renames" — 20+ major variables (wages, interest, dividends, pensions, Social Security, etc.) each get their own SOI-based per-return growth factor. Itemized deductions are overridden with a fixed 2% annual rate.

- **Block 2 (lines 156-162)**: "Pos/neg split" — 3 variables where positive and negative values are scaled by different SOI growth factors: E00900 (Sch C business income), E01000 (capital gains), and E26270 (partnership/S-corp income).

- **Block 3 (lines 167-169)**: "Remaining" — 42 other variables that all get AGI per-return growth as a catch-all fallback.

Each block runs sequentially on the same dataframe, so a variable appearing in multiple blocks gets scaled multiple times.

## The bug

E00900 and E26270 appear in **both** Block 2 (pos/neg split) **and** Block 3 (remaining variables). They are scaled twice: first by their variable-specific SOI growth, then again by AGI growth.

E01000 (capital gains), the third pos/neg split variable, is correctly excluded from `REMAINING_VARIABLES` and is only scaled once.

### Effective scaling (2015 to 2022)

| Variable | Block 2 factor (pos) | Block 2 factor (neg) | Block 3 factor (AGI) | Combined (pos) | Combined (neg) |
|----------|--------------------:|--------------------:|--------------------:|---------------:|---------------:|
| E00900   | 1.294               | 2.065               | 1.355               | **1.754**      | **2.799**      |
| E26270   | 1.605               | 1.979               | 1.355               | **2.175**      | **2.682**      |

Block 2 alone correctly applies the SOI-implied per-return growth. The bug is that Block 3 then multiplies again by AGI per-return growth (1.355), making each record's value 35.5% larger than SOI implies:

| Variable | SOI-implied growth (Block 2 alone) | Actual applied (Block 2 × Block 3) |
|----------|:----------------------------------:|:-----------------------------------:|
| E00900 positive | +29.4% (correct: $392B → $544B) | **+75.3%** (overstated) |
| E00900 negative | +106.5% (correct: $60B → $133B) | **+179.8%** (overstated) |
| E26270 positive | +60.5% (correct: $756B → $1,300B) | **+117.5%** (overstated) |
| E26270 negative | +97.9% (correct: $127B → $269B) | **+168.2%** (overstated) |

These are large variables — E26270 alone is $1.3T in 2022, and E00900 profits are $544B. The double-scaling overstates them by 35.5%, meaning the reweighter must compensate for hundreds of billions in excess amounts.

### Origin

Both the pos/neg split logic and the `REMAINING_VARIABLES` list (which includes E00900 and E26270) were created in the same commit by Nikhil Woodruff: `f92f173` (June 17, 2024, "Add better SOI replication"). The original code has this inline comment on the REMAINING_VARIABLES block:

```python
# Remaining variables, uprate purely by AGI growth (for now, because I'm not sure
# how to handle the deductions, credits and incomes separately)
```

This suggests the REMAINING_VARIABLES list was intended as a catch-all for variables without specific SOI targets. E00900 and E26270 were likely included before the pos/neg split logic was added (or without realizing they'd be handled twice). E01000 (capital gains), the third pos/neg split variable, was correctly excluded from REMAINING_VARIABLES.

### Impact

The reweighting step (Clarabel QP) likely compensates by adjusting weights to hit SOI cell targets. However:

- The overstated amounts force the reweighter to work harder (push weights further from 1.0)
- Records with large E00900 or E26270 values may get systematically under-weighted
- This could distort the distribution of these variables even if the aggregate target is met

### Suggested fix

Remove E00900 and E26270 from `REMAINING_VARIABLES`:

```python
REMAINING_VARIABLES = [
    "E03500",
    "E00800",
    ...
    # "E26270",  # REMOVE — already in SOI_TO_PUF_POS/NEG_ONLY_RENAMES
    ...
    # "E00900",  # REMOVE — already in SOI_TO_PUF_POS/NEG_ONLY_RENAMES
    ...
]
```

### Questions before fixing

1. Should we regenerate the TMD file after fixing and compare results?
2. Should we add a test that checks for variables appearing in multiple scaling lists?
3. Is the current reweighted output "good enough" despite the bug, or does fixing this meaningfully improve quality?
