# Email to SOI: A59664 Unit Error in Congressional District Data

**To:** IRS Statistics of Income Division
**Subject:** Possible unit error in column A59664 of 2022 Congressional District data file (22incd.csv)

---

Dear SOI team,

I believe I've found a unit error in the 2022 Congressional District
individual income tax data file (22incd.csv), available at
https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi.

## The issue

Column **A59664** (Earned income credit amount for returns with three
or more qualifying children) appears to be published in **dollars**
rather than **thousands of dollars**.  All other EITC amount columns
in the same file (A59660, A59661, A59662, A59663) are correctly in
thousands of dollars, consistent with the documentation.

## Evidence

At the US aggregate level (summing all areas at AGI_STUB = 0):

| Column | Description | Value |
|--------|-------------|-------|
| A59660 | Total EITC amount | 58,124,026 ($1000s) |
| A59661 | EITC, no children | 2,074,476 ($1000s) |
| A59662 | EITC, 1 child | 15,671,458 ($1000s) |
| A59663 | EITC, 2 children | 26,751,351 ($1000s) |
| A59664 | EITC, 3+ children | **13,600,954,503** |

The sum A59661 + A59662 + A59663 + A59664 = 13,645,451,788, which
is ~235 times A59660.  This is inconsistent.

However, A59661 + A59662 + A59663 + (A59664 / 1000) = 58,097,240,
which matches A59660 within 0.05% (consistent with normal SOI cell
suppression/rounding).

For comparison, the **2022 state-level file** (22in55cmcsv.csv) has
A59664 = 13,861,484, correctly in thousands of dollars, and the
subcategory columns sum to within 4 of A59660.

The count columns (N59660 through N59664) appear correct in both
files -- this issue affects only the **amount** column A59664.

## Scope

- **Affected:** 2022 Congressional District file (22incd.csv), column A59664
- **Not affected:** 2022 State file (22in55cmcsv.csv) -- A59664 is correct
- **Unknown:** County file, other years -- I have not checked these

The error is consistent across all 428 congressional districts and
all 9 AGI stubs in the file.

## Verification steps

To confirm this finding:
1. Sum A59661 + A59662 + A59663 + A59664 at AGI_STUB = 0 for any state
2. Compare to A59660 for the same state/stub
3. The sum will overshoot by approximately 1000x
4. Dividing A59664 by 1000 before summing produces a match

Thank you for your work maintaining these valuable datasets.  Please
let me know if you need any additional information.

Best regards,
[Your name]
