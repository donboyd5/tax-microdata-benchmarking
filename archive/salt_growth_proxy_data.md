# SALT Growth Proxy: Census State and Local Government Finance Data

## Summary

We can improve upon the 2% per-return annual growth rate currently used for SALT deductions  (E18400, E18500) during PUF uprating from 2015 to 2022. Instead, we can use a proxy for SALT growth -- the actual growth in state and local taxes collected per return, or approximately 4.7% per year.

## Background

Beginning in 2018, the 2017 TCJA raised the standard deduction and capped some itemized deductions -- particularly the state and local tax (SALT) deduction. This change sharply reduced the number of itemizers and the amount of itemized deductions, especially for SALT.

TMD currently addresses this by "growing" underlying itemized deductions in the 2015 PUF to our 2022 base year at 2% per year. Growing these deductions made sense, essentially treating them as *available* deductions that Tax-Calculator would then apply relevant tax rules to. This allows policy analyses that might involve different standard deduction amounts or different itemized deduction caps - analyses that would not be possible if we targeted actual (capped) deductions of actual itemizers in 2022 or later years. (In a perfect world, we would construct estimates of available deductions for all taxpayers, not just 2015 itemizers. That is not practical at present. However, 2015 itemizers are a considerably larger universe than current itemizers and this is a good second-best.)

The 2% assumption was found to work well for targeting tax liability, based on empirical analysis at the time.

We now have more IRS data than when the 2% rule was decided, and more important, we have more external data that can serve as proxies for how available SALT deductions might have evolved. Available SALT deductions likely have grown considerably faster. Preliminary review of potential proxies for other itemized deductions, which are smaller than SALT, do not yet suggest any changes to the 2% rule for them.

## Source

**U.S. Census Bureau, Annual Survey of State and Local Government Finances**

Table 1: State and Local Government Finances by Level of Government and by State

- **2015**: "2015 Annual Surveys of State and Local Government Finances"
  - File: `15slsstab1a.xlsx`
  - URL: https://www2.census.gov/programs-surveys/gov-finances/tables/2015/summary-tables/15slsstab1a.xlsx

- **2022**: "2022 Census of Governments: Finance"
  - File: `22slsstab1.xlsx`
  - URL: https://www2.census.gov/programs-surveys/gov-finances/tables/2022/22slsstab1.xlsx

Both files report "United States Total" for "State & local government" combined. Dollar amounts are in thousands.

Note: 2015 is from the Annual Survey (sample-based for local governments). 2022 is from the Census of Governments (full enumeration, conducted every 5 years).

## Data

All amounts are State & Local government combined, U.S. total. Dollar amounts in Census files are in thousands; shown here in billions. Growth rates are computed from unrounded thousands.

| Item | Line | 2015 ($B) | 2022 ($B) | Growth |
|------|-----:|----------:|----------:|-------:|
| Total taxes | 8 | $1,563.7 | $2,367.7 | +51.4% |
| Property taxes | 9 | $484.3 | $649.0 | +34.0% |
| General sales taxes | 11 | $368.0 | $557.2 | +51.3% |
| Individual income taxes | 18 | $368.9 | $600.6 | +62.8% |

## SALT proxy construction

E18400 covers "state and local income **or** sales taxes" (taxpayers choose one). E18500 covers real estate (property) taxes. The relevant proxy combines all three tax types:

| Component | 2015 ($B) | 2022 ($B) |
|-----------|----------:|----------:|
| Property taxes | $484.3 | $649.0 |
| General sales taxes | $368.0 | $557.2 |
| Individual income taxes | $368.9 | $600.6 |
| **SALT proxy total** | **$1,221.2** | **$1,806.8** |

**Total SALT proxy growth: 1.4796 (+48.0%)**

Per-return growth (dividing by IRS returns growth of 1.0721): **1.3802 (+38.0%)**

Implied annual rate: **4.72%/year**

## Comparison to current assumption

| | Per-return growth | Annual rate |
|--|------------------:|------------:|
| Current (ITMDED_GROW_RATE = 0.02) | +14.9% | 2.0% |
| Census SALT proxy | +38.0% | 4.7% |

The current 2%/year rate understates SALT growth considerably.

## Caveats

1. The Census data includes taxes paid by businesses, not just individuals. The SALT deduction is for taxes paid by individuals only. Business property taxes and sales taxes are included in the Census totals but are not deductible on individual returns.

2. The proxy applies a single combined growth rate to both E18400 and E18500. In practice, property taxes grew more slowly (+34.0%) than income/sales taxes (+57.0%). A future refinement could apply separate rates.

3. The Census fiscal years don't perfectly align with IRS tax years, but they are close enough for this purpose.

## Recommendation

Replace the current single 2%/year SALT growth rate with a Census-based rate derived from property + general sales + individual income tax growth: approximately **4.7%/year** (equivalent to +48.0% total or +38.0% per-return over 7 years).