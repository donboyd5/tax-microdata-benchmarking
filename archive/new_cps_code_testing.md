# New CPS Code Testing Notes
*Branch: `new-cps-code`*
*Date: 2026-03-10*

---

## Test: Upstream commit 16d41f19

**Commit:** `16d41f1970a3d7d2599f038cf467eac6a19d8424`
**Message:** "Add _is_tax_filer function in tmd/datasets/cps.py module"

Reset local `new-cps-code` to upstream `new-cps-code` at this commit.

### Build Results

`make clean && make data` — **success**
- 51 passed, 3 skipped
- Clarabel solver: 9.8s, 19 iterations, all 550 targets within 0.5% tolerance

### CPS Nonfiler Comparison

| | **New TMD** (new-cps-code @ 16d41f19) | **Master TMD** |
|---|---|---|
| **Unweighted CPS records** | 12,269 | 17,564 |
| **Weighted CPS count** | 23,565,409 | 22,278,615 |
| **Total TMD records** | 219,961 | 225,256 |

### Observations

- The new `_is_tax_filer` function drops ~5,300 CPS records (from 17,564 to 12,269)
  - Presumably reclassifies some CPS tax units as filers and excludes them from the nonfiler pool
- Despite fewer records, weighted nonfiler population *increases* slightly: 22.3M → 23.6M (+1.3M, +5.8%)
  - Likely due to CPS_WEIGHTS_SCALE reweighting and/or different weight distribution among remaining records
- Total TMD records decreased from 225,256 to 219,961 (-5,295), matching the CPS record drop
- PUF records unchanged at 207,692
