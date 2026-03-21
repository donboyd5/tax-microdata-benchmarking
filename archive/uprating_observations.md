# PUF Uprating Observations (2015 → 2022)

## How uprating works

`tmd/datasets/uprate_puf.py` uses `get_growth()` which pulls SOI aggregates from `tmd/storage/input/soi.csv` to compute per-return growth factors:

```
per_ret_growth = (soi_2022_amount / soi_2015_amount) / (soi_2022_returns / soi_2015_returns)
```

Weights (s006) are scaled separately by `soi_2022_returns / soi_2015_returns`.

Growth is applied in three blocks:
- **Block 1 (straight renames)**: Variable-specific SOI growth (or fixed 2% for itemized deductions)
- **Block 2 (pos/neg split)**: Separate SOI growth for positive vs negative values
- **Block 3 (remaining)**: All get AGI per-return growth as fallback

## Key growth factors (2015 → 2022)

| Item | Per-return growth | Weight growth | Total growth |
|------|------------------:|--------------:|-------------:|
| # Returns | — | +7.2% | +7.2% |
| Wages (E00200) | +27.7% | +7.2% | +36.9% |
| Taxable interest (E00300) | +30.0% | +7.2% | +39.3% |
| Ordinary dividends (E00600) | +47.8% | +7.2% | +58.4% |
| Capital gains gross (E01000+) | +66.0% | +7.2% | +77.9% |
| Partnership/S-corp income (E26270+) | +60.5% | +7.2% | +72.0% |
| Total pensions (E01500) | +22.0% | +7.2% | +30.7% |
| Social Security (E02400) | +34.3% | +7.2% | +44.0% |
| Unemployment comp (E02300) | +3.6% | +7.2% | +11.1% |
| **AGI (fallback for remaining vars)** | **+35.5%** | **+7.2%** | **+45.3%** |

For Block 1 "straight rename" variables, total growth algebraically equals the SOI aggregate ratio by construction (the returns ratio cancels). This is a tautology, not something that needs testing.

For Block 2 "pos/neg split" variables, positive and negative records get different growth factors, so the net variable won't exactly match any single SOI number.

## The 42 REMAINING_VARIABLES — all get AGI growth

All variables in `REMAINING_VARIABLES` receive AGI per-return growth (+35.5%). They have no variable-specific SOI calibration and are not targeted by the reweighter. Their aggregates in the final TMD are essentially uncontrolled.

Sources: 2015 PUF General Description Booklet (IRS/SOI); Tax-Calculator `input_vars.md`.

### Income

| PUF Name | Description |
|----------|-------------|
| E00700 | Taxable refunds of state and local income taxes |
| E00800 | Alimony received |
| E00900 | Business or profession net profit/loss (Sch C) — **also in pos/neg split lists, see double-scaling issue** |
| E01200 | Other gains or loss (Form 4797) |
| E25850 | Rent/royalty net income (Sch E) |
| E25860 | Rent/royalty net loss (Sch E) |
| E26270 | Combined partnership and S-corp net income/loss (Sch E) — **also in pos/neg split lists, see double-scaling issue** |
| E26390 | Estate/trust total income (Sch E) |
| E26400 | Estate/trust total loss (Sch E) |
| E27200 | Farm rent net income or loss (Sch E) |
| T27800 | Elected farm income (Sch J) |

### Capital gains (Sch D detail)

| PUF Name | Description |
|----------|-------------|
| P22250 | Short-term capital gains less losses net of carryover |
| P23250 | Long-term capital gains less losses net of carryover |
| E24515 | Unrecaptured Section 1250 gain |
| E24518 | 28% rate gain or loss |

Note: Gross capital gains (E01000 positive) ARE targeted in Block 2. These Sch D detail components get AGI growth instead of capital-gains-specific growth.

### Statutory adjustments (above-the-line deductions)

| PUF Name | Description |
|----------|-------------|
| E03150 | Total deductible IRA contributions |
| E03210 | Student loan interest deduction |
| E03220 | Educator expenses |
| E03230 | Tuition and fees deduction (Form 8917) |
| E03240 | Domestic production activities deduction (Form 8903) — repealed by TCJA |
| E03270 | Self-employed health insurance deduction |
| E03290 | Health savings account deduction (Form 8889) |
| E03300 | Payments to Keogh / SEP / SIMPLE / qualified plans |
| E03400 | Forfeited interest penalty (early withdrawal of savings) |
| E03500 | Alimony paid |

### Itemized deductions (Sch A)

| PUF Name | Description |
|----------|-------------|
| E20100 | Other than cash charitable contributions |
| E20400 | Miscellaneous deductions subject to AGI limitation — zero post-TCJA |
| E20500 | Net casualty or theft loss — essentially zero post-TCJA |

### Credits

| PUF Name | Description |
|----------|-------------|
| E07240 | Retirement savings contributions credit (Form 8880) |
| E07260 | Residential energy credit (Form 5695) |
| E07300 | Foreign tax credit (Form 1116) |
| E07400 | General business credit (Form 3800) |
| E07600 | Prior year minimum tax credit (Form 8801) |
| E62900 | Alternative minimum tax foreign tax credit (Form 6251) |
| E87521 | American Opportunity Credit |
| P08000 | Other tax credits (not including Sch R credit) |

Note: Scaling credits by AGI growth is conceptually odd — credit amounts don't typically grow proportional to income — but most are small.

### Other

| PUF Name | Description |
|----------|-------------|
| E32800 | Child/dependent-care expenses for qualifying persons (Form 2441) |
| E58990 | Investment income elected amount (Form 4952) |
| E09700 | Recapture of investment credit |
| E09800 | Unreported payroll taxes (Form 4137 or 8919) |
| E09900 | Penalty tax on qualified retirement plans |
| E11200 | Excess FICA/RRTA tax withheld |

### Inconsistencies worth noting

- **Rent/royalty (E25850/E25860)** and **estate/trust (E26390/E26400)**: SOI has variable-specific 2015 aggregates (`rent_and_royalty_net_income`: $103.1B, `estate_income`: $32.5B) that *could* be used for targeted growth, but aren't.
- **Sch D capital gains (P22250, P23250)**: Gross E01000 IS targeted in Block 2, but these detail components get AGI growth instead of capital-gains-specific growth.
- **Post-TCJA zeroes**: E20400 (misc deductions), E20500 (casualty loss), E03240 (domestic production) are effectively zero after TCJA. Scaling by AGI growth is harmless but meaningless.

## Itemized deductions: fixed 2% growth vs SOI reality

Block 1 overrides SOI-based growth for five itemized deduction variables with a fixed 2% annual rate (`ITMDED_GROW_RATE = 0.02` in `imputation_assumptions.py`). Over 7 years this gives a factor of 1.149 (+14.9% per return).

| Item | 2015 SOI ($B) | 2022 SOI ($B) | Actual growth | Applied growth |
|------|-------------:|-------------:|--------------:|---------------:|
| Medical (E17500) | $133.8B | $121.0B | -9.6% | +14.9% |
| State income tax (E18400) | $335.1B | $257.4B | -23.2% | +14.9% |
| Real estate tax (E18500) | $188.6B | $107.3B | -43.1% | +14.9% |
| Interest paid (E19200) | $304.5B | $170.5B | -44.0% | +14.9% |
| Charitable cash (E19800) | $221.9B | $222.4B | +0.2% | +14.9% |

The actual SOI shows large declines driven by TCJA's $10K SALT cap reducing itemization. The 2% assumption ignores TCJA effects. The reweighter must compensate for this mismatch.

This may be intentional — the reasoning might be that uprating individual record amounts by the SOI aggregate decline would incorrectly shrink itemizers' deduction amounts, when really the change was driven by people *switching* from itemizing to the standard deduction. The reweighter can then handle this by shifting weight away from itemizers. But this deserves confirmation.