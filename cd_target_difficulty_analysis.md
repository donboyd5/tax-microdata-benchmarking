# CD Target Difficulty Analysis

How different congressional districts challenge the area weighting
optimizer, and what this means for recipe design.

## The Core Metric: Gap from Proportionate Share

For each target, we compute what the area would get under simple
population-proportional allocation (`pop_share * national_total`)
and compare it to the actual target (from SOI geographic shares).

A 0% gap means the area looks just like the nation for that
variable/bin.  A +50% gap means the area needs 50% more than its
population share.  A -50% gap means it needs half as much.

```bash
python -m tmd.areas.developer_mode --difficulty AL01
python -m tmd.areas.developer_mode --difficulty NY12
```

## Four Representative CDs

| CD | Description | Pop share | Mean gap | Median | Max | Easy | Moderate | Hard | Very hard |
|----|-------------|-----------|---------|--------|-----|------|----------|------|-----------|
| AL-01 | Rural South, retirees | 0.215% | 29% | 23% | 199% | 16 | 31 | 41 | 18 |
| MN-03 | Suburban Minneapolis | 0.214% | 48% | 28% | 783% | 8 | 29 | 27 | 30 |
| TX-20 | South Texas border | 0.230% | 38% | 41% | 98% | 9 | 15 | 37 | 33 |
| NY-12 | Manhattan | 0.233% | 255% | 55% | 3384% | 2 | 12 | 28 | 52 |

(Easy: <5%, Moderate: 5-20%, Hard: 20-50%, Very hard: >50%.
95 targets in current spec.)

## AL-01: Alabama First District

**Character:** Rural/suburban district anchored by Mobile.  More
retirees, fewer high earners than the national average.

**Key gaps (most difficult targets):**
- `e26270 amt $10K-$25K`: **-199%** (partnership income flips sign)
- `e18500 amt all` (SALT real estate): **-81%** (low property values)
- `e00300 amt $500K+` (interest, high-AGI): **-68%** (few wealthy)
- `e00200 amt $500K+` (wages, high-AGI): **-59%** (few high earners)
- `eitc amt all`: **+61%** (more EITC recipients than average)
- `e02400 amt $10K-$25K` (SS): **+32%** (retirees)

**Pattern:** Almost everything related to wealth is below
proportionate (negative gap), while retirement income and credits
are above.  The solver downweights high-income records and
upweights retiree/low-income records.  Moderate difficulty overall.

## NY-12: Manhattan

**Character:** Extreme wealth concentration.  Wall Street, Upper
East Side.  More single filers, very high incomes, massive capital
gains, charitable giving, and partnership income.

**Key gaps (most extreme in the nation):**
- `e26270 amt $10K-$25K` (partnerships): **+3,384%** (need 34x proportionate)
- `e00300 amt <$0K` (interest, neg-AGI): **+2,796%**
- `e00300 amt $500K+` (interest): **+1,718%**
- `e18400 amt all` (SALT income/sales): **+1,218%**
- `capgains_net amt $500K+`: **+898%** (10x proportionate cap gains)
- `e00200 amt $500K+` (wages): **+809%**
- `c00100 amt $500K+` (AGI): **+797%**

**Pattern:** Every high-income variable is dramatically above
proportionate.  The solver must push x[i] to 10-30x for records
with Manhattan-like profiles.  This is why the multiplier cap
matters — 50x barely suffices for the most extreme records.

Only 2 of 94 targets are "easy" (<5% gap).  Developer mode drops
several targets that simply can't be hit within bounds.

## TX-20: South Texas Border District

**Character:** Low-income, heavily Hispanic, many EITC recipients.
Few high earners or investors.

**Key gaps:**
- `e26270 amt $0K-$10K` (partnerships): **-98%** (almost none)
- `e00300 amt $500K+` (interest): **-86%**
- `eitc amt all`: **+82%** (heavy EITC usage)
- `e00200 amt $500K+` (wages): **-73%** (few high earners)
- `capgains_net amt $500K+`: **-66%**
- `e02400 amt $0K-$10K` (SS): **+52%** (retirees)
- `c00100 returns all` (total returns): **+42%** (more returns per capita)

**Pattern:** Mirror image of Manhattan — everything related to
wealth is sharply negative, credits and retirement income are
positive.  More uniformly difficult than AL-01 (33 targets
>50% vs 18), but without Manhattan's extreme outliers.

## MN-03: Suburban Minneapolis

**Character:** Affluent suburban district.  Professional incomes,
good property values, some high earners but not Manhattan-extreme.

**Key gaps:**
- `e26270 amt $10K-$25K` (partnerships): **+783%** (strong partnership presence)
- `c19200 amt all` (mortgage interest): **+64%** (homeowners)
- `e00200 amt $500K+` (wages): **+55%** (professional earners)
- `capgains_net amt $500K+`: **+44%**
- `eitc amt all`: **-45%** (fewer EITC recipients)
- `e02400 amt $10K-$25K` (SS): **-37%** (younger population)

**Pattern:** Above-average but not extreme wealth.  The partnership
outlier (`e26270 $10K-$25K` at +783%) is a single problematic bin
where the SOI target and the national microdata's bin composition
diverge sharply.

## Lessons for Recipe Design

### 1. Every CD is different, but patterns cluster

Rural/retiree CDs (AL-01 type): wealth targets negative, credits
positive.  Urban wealthy CDs (NY-12 type): everything extreme.
Suburban CDs (MN-03 type): moderately above average.  Border/
low-income CDs (TX-20 type): opposite of wealthy CDs.

### 2. The $500K+ bin is universally the hardest

Across all CDs, the highest-AGI bin shows the largest gaps because
high-income filers are geographically concentrated.  A CD either
has a lot of them (Manhattan: +800%) or very few (South Texas: -70%).

### 3. Partnership income is wildly variable by geography

`e26270` in low AGI bins produces the most extreme individual gaps
(3,384% for NY-12).  This is because partnership losses/income are
concentrated in specific financial centers.  Consider making
partnership targets total-only in low-AGI bins.

### 4. Credits need total-only targets

Per-bin EITC/CTC targets require the solver to match both income
distribution AND credit eligibility per bin — conflicting constraints
that cause solve time explosions (12s → 92s).  Total-only credit
targets are sufficient and dramatically cheaper.

### 5. Difficulty predicts solve time

Areas with higher mean |gap| take longer to solve and are more
likely to need overrides.  This suggests developer mode could
use the difficulty table to predict which areas need special
attention before even running the solver.

### 6. The multiplier cap constrains extreme areas

For NY-12, the solver needs some records at 30x+ their proportionate
weight.  The 50x cap is binding.  For typical CDs like AL-01, most
multipliers stay in 0.5–2.0 range and the cap is irrelevant.
