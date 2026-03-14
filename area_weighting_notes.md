**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Area weighting

---


## Goals

- Extend area data used to create area weights for states and for Congressional districts to 2022; we already have 2021
- Make the process of developing area targets and mapping them to TMD variables smoother and cleaner
- Ensure that either 2021 or 2022 data can be used to estimate each area's share of national data for specific income and other items: 2021 data shares can be used for any national data year, 2022 area data shares can be used for any national data year, and so on. In practice, we'll almost always pair 2021 area data with 2021 national data and 2022 area data with 2022 national data. But the flexibility to use different years is important.
- Make sure we can reproduce files created under the old process with files under the new process
- Make the area weighting optimization considerably faster. Consider converting the objective function to minimizing change from average national weights, subject to constraints, much as we did with national reweighting. This could involve using the Clarabel solver.
- Look for ways to create area weights en masse efficiently -- all 50 states, or all 435 Congressional districts - perhaps benefiting from solver features and also paralellism
- Convert the data preparation from R to python

## Key Design Principles

### A. All targets as shares (eventual goal)
Area targets = national TMD total × (area SOI / national SOI).
SOI provides geographic distribution, TMD provides levels.
Shares computed at variable × AGI-bin × filing-status granularity.
Transition: reproduce R pipeline first, then convert all to shares.

### B. Optimization variable: deviation from population-proportional
x_i = area_weight / (pop_share × national_weight).
x_i = 1 means population-proportional share.
QP minimizes sum((x_i - 1)²) + M·sum(s_j²) subject to target bounds with elastic slack s_j >= 0 (penalty M=1e6) and multiplier bounds [x_min, x_max] (default [0.0, 100.0]).

## Progress

### Completed

**Phase 1: Module structure and constants** (2026-03-12)
- Created `tmd/areas/prepare/` package with `__init__.py` and `constants.py`
- Constants: AreaType enum, AGI cuts (10 state bins, 9 CD bins), SOI file patterns (2015-2022), SHARING_MAPPINGS, STATE_INFO, ALLCOUNT_VARS

**Phase 2: Clarabel area weight optimizer** (2026-03-12)
- Created `tmd/areas/create_area_weights_clarabel.py` (~620 lines)
- Constrained QP mirroring national reweighting: minimize sum((x_i-1)²) with elastic slack
- Default params: tolerance=0.004, multiplier bounds [0.0, 100.0], slack penalty=1e6
- Bridge flag `USE_CLARABEL = True` in `create_area_weights.py`
- Performance on "xx" area: Clarabel ~9s (all 16 targets hit), L-BFGS-B ~7.5s (1 target miss)
- All 50 existing tests pass

**Phase 3: State SOI data ingestion** (2026-03-12)
- Created `tmd/areas/prepare/soi_state_data.py` — reads raw SOI CSVs, pivots to long format, classifies variables, creates derived vars (18400 = 18425 + 18450), scales amounts x1000, adds AGI labels, produces base_targets
- Created `tmd/areas/prepare/census_population.py` — embedded 2021 Census PEP state populations + ACS 1-year CD populations (436 CDs + US)
- Tested: 8 years (2015-2022) loaded, ~98K rows per year, all 54 areas (incl OA), verified NY values

**Phase 4: Target file writer** (2026-03-12)
- Created `tmd/areas/prepare/target_file_writer.py` — JSON recipe parser (strips // comments), variable mapping loader, match frame builder, per-area CSV writer
- Produces output matching format expected by `create_area_weights.py`: varname,count,scope,agilo,agihi,fstatus,target
- Tested: generates 144-147 targets per state across all 53 areas

**Phase 6a: Target sharing pipeline** (2026-03-12)
- Created `tmd/areas/prepare/target_sharing.py` — computes TMD national sums by AGI bin, SOI geographic shares, derives area targets = TMD_sum x SOI_share
- 4 shared variables: e01500<-01700, e02400<-02500, e18400<-18400, e18500<-18500
- `build_enhanced_targets()` combines base + shared targets with sort ordering
- Full pipeline tested end-to-end: SOI -> base_targets -> enhanced_targets -> target files for all 53 areas

**Phase 5: CD SOI data ingestion** (2026-03-12)
- Created `tmd/areas/prepare/soi_cd_data.py` — reads SOI CD CSV from ZIP, classifies record types (US, DC, cdstate, state, cd), pivots to long, creates derived vars, adds AGI labels, builds base_targets with area codes
- Created `tmd/areas/prepare/cd_crosswalk.py` — 117th-to-118th Congress boundary crosswalk module using geocorr data (ready for crosswalk file when provided)
- Updated `census_population.py` with embedded 2021 ACS 1-year CD populations (436 CDs + US total)
- Key finding: BOTH 2021 and 2022 SOI CD data use 117th Congress boundaries. MT=1 CD (not 2), OR=5 CDs (not 6), TX=36 (not 38) in both years. Crosswalk needed for both years to get 118th Congress targets.
- Tested: 2021 data produces 437 areas (436 CDs + US), 721K base_target rows. 2022 data also reads successfully.

**Phase 6b: All-shares targets** (2026-03-12)
- Added `ALL_SHARING_MAPPINGS` to constants.py — 15 variable combos (8 amounts, 3 nonzero counts, 4 allcounts by filing status)
- New `compute_tmd_national_sums_all()` handles count types 0 (amounts), 1 (allcounts via MARS), 2 (nonzero counts)
- New `build_all_shares_targets()` replaces ALL direct SOI values with TMD x SOI_share
- Created allshares variable mapping CSVs for both states and CDs
- Legacy 4-var sharing preserved via `build_enhanced_targets()` for backward compatibility
- Tested: MN produces 166 all-shares targets (same as legacy target count when processed through the target file writer)
- All-shares and legacy target different VALUES but the same target variables/concepts; mean difference ~8% (SOI vs TMD national differences)

**Phase 7: 2022 data extension** (2026-03-12)
- Moved population data from embedded Python dicts to JSON files in `tmd/areas/prepare/data/`
- Added 2022 state populations (PEP vintage 2023, from NST-EST2023-ALLDATA.csv)
- Added 2022 CD populations (ACS 2022 1-year, 118th Congress boundaries, 436 CDs)
- Key: 2022 ACS CD data is on 118th Congress boundaries, but SOI CD data (both 2021 and 2022) is on 117th — so 2021 CD populations used for 117th Congress processing regardless of SOI year
- Tested: full 2022 state pipeline (SOI → base_targets → all-shares) produces 166 targets per state
- Tested: full 2022 CD pipeline produces 151 targets per CD (NY01 example)
- All 50 existing tests pass

**Phase 8: Flexible year pairing** (2026-03-12)
- Added `prepare_area_targets()` orchestrator in `target_sharing.py`
- Supports independent `area_data_year`, `national_data_year`, `pop_year`
- Tested: 2022 SOI shares + 2021 TMD nationals → ~2% geographic shift for MN AGI (expected)
- All 50 existing tests pass

### Performance (measured, Clarabel solver with all-shares targets)
- ~212,000 TMD records per area
- **Per state: 25-35 seconds** (107 targets), ~30s average
- **DC: 203 seconds** (InsufficientProgress, 13 violated targets — special case)
- **50 states + DC + xx = 52 areas with 8 workers: 3.7 minutes total**
- **Estimated 435 CDs with 8 workers: ~35 minutes**
- **Estimated all 485 areas with 8 workers: ~40 minutes**
- 13 of 52 states had some violated targets (mostly small states: AK, ND, SD, VT, WV, WY)

**Phase 9: Batch processing** (2026-03-12)
- Created `tmd/areas/batch_weights.py` with ProcessPoolExecutor
- TMD data loaded once per worker process (not once per area) via `_init_worker()`
- Progress reporting with ETA, violated target tracking
- Area filtering: `--areas states`, `--areas cds`, `--areas all`, or comma-separated
- `--force` flag to recompute all areas even if up-to-date
- Usage: `python -m tmd.areas.batch_weights --workers 8 --areas states --force`
- All 50 existing tests pass

**Phase 10: Validation** (2026-03-12)
- Fixed bug in `target_file_writer.py`: allcount variable filter didn't recognize shared basesoivnames (e.g., `tmd00100_shared_by_soin1` vs `n1`). This caused all count=1 targets (40 per state) to be dropped with allshares mapping. Fixed by extracting SOI part from shared names.
- **Target count validation**: Old and new pipelines produce IDENTICAL target counts:
  - MN state: 147 targets (both old and new)
  - MN01 CD: 128 targets (both old and new)
- **Target value comparison (MN state)**:
  - Previously shared variables (e01500, e02400, e18400, e18500): typically <1% difference, max ~10% for lowest AGI bins of SALT variables
  - Newly shared variables (c00100, e00200, e00300): mostly 1-5% difference (TMD national differs from SOI)
  - e26270 (partnership/S-corp): up to 445% in small AGI bins — expected, volatile variable
  - c00100 counts (lowest AGI bin): 20-40% — TMD return counts differ from SOI by bin
- **Target value comparison (MN01 CD)**:
  - Larger differences overall (3-28%) because old CD targets went through 117th→118th Congress crosswalk
  - XTOT population: 3.24% difference (different Census source)
  - e18400/e18500: 10-28% in some bins
- **Solver comparison**: Both MN and MN01 solved successfully with Clarabel:
  - MN: 146/147 targets hit, 12.4s, multiplier median=0.994, RMSE=0.317
  - MN01: 124/128 targets hit, 14.1s, multiplier median=0.930, RMSE=0.463
- **Weight comparison**: Can't match record-by-record (old TMD had 225K records vs new 212K). Aggregate comparison: weight sums differ by 3-5% (different TMD base), similar distributions. Clarabel produces more zero weights (1.5% state, 7.5% CD) vs old solver (0.1%).

**Phase 11: CD crosswalk (117th→118th Congress)** (2026-03-12)
- User created new Geocorr 2022 crosswalk from MCDC (https://mcdc.missouri.edu/applications/geocorr2022.html): source=118th Congress, target=117th Congress, population-weighted, 0-weighted blocks ignored
- Crosswalk CSV saved to `tmd/areas/prepare/data/geocorr2022_cd117_to_cd118.csv` (1,447 rows including labels)
- Existing `load_geocorr_crosswalk()` worked with new data: cleans PR, DC98→DC00, pads NC codes, computes pop-weighted shares
- `allocate_117_to_118()` correctly allocates: MT00→{MT01,MT02}, OR 5→6 CDs, CO 7→8, TX 36→38
- Allocation factor sums verified: all sum to exactly 1.0 per cd117
- Target conservation verified: MN totals identical before/after crosswalk (0.000000% diff)
- Integrated into `prepare_area_targets()`: new `apply_cd_crosswalk` (default True) and `crosswalk_path` parameters
- Result: 436 areas on 118th Congress boundaries (435 CDs + DC)
- Fixed incorrect docstring in `cd_crosswalk.py` that claimed 2022 data was on 118th boundaries
- All 50 existing tests pass

**Phase 12: End-to-end orchestration** (2026-03-12)
- Created `tmd/areas/prepare_and_solve.py` — single CLI for full pipeline
- Usage: `python -m tmd.areas.prepare_and_solve --scope states --workers 8`
- Three stages: `targets` (prepare+write), `solve` (Clarabel), or `all` (both)
- Scopes: `states`, `cds`, `all`, or comma-separated area codes (e.g., `MN,MN01`)
- Supports `--year`, `--national-year`, `--pop-year` for flexible year pairing
- **All 52 state areas tested** (8 workers, 5.3 min total):
  - 47 solved; 22 with 0 violated targets
  - 4 PrimalInfeasible (CA, DC, MD, UT, VA) — needs parameter tuning
  - 1 InsufficientProgress (GA, 287s, 6 violated)
- **CD pipeline tested**: MN01, MN02, TX37, OR06 (new 118th districts) — all solved
- All 50 existing tests pass

**Phase 13: Variable alignment analysis and safe recipe** (2026-03-12)

Investigated why 5 states (CA, DC, MD, UT, VA) returned PrimalInfeasible with the full 147-target recipe. Root cause: badly misaligned SOI↔TMD variable definitions for several targets.

**SOI vs TMD national total comparison** (% diff = (TMD/SOI - 1) × 100):

Well-aligned (<2%):
- c00100 (AGI): +0.5%
- e00200 (wages): -0.3%
- e00300 (interest): -0.0%
- e26270 (partnership/S-corp): +0.0%
- Return counts (n1, mars1, mars2, mars4): +0.8% to +1.8%
- e00200 nonzero count: +0.6%

Badly misaligned (>10%):
- e01500 (TMD total pensions) vs a01700 (SOI taxable pensions): +76.7%
- e02400 (TMD total SS) vs a02500 (SOI taxable SS): +93.2%
- e18400 (TMD SALT all filers) vs a18400 (SOI SALT itemizers only): +70.0%
- e18500 (TMD SALT RE all filers) vs a18500 (SOI SALT RE itemizers only): +125.3%
- e18400 nonzero count: +211.8%, e18500 nonzero count: +208.4%

**SOI does NOT have total pension (a01500) or total SS (a02400) variables** — only taxable versions (a01700, a02500). Cannot fix alignment by switching SOI variables.

**"Safe" minimal recipe** created with only well-aligned variables:
- Files: `states_safe.json`, `state_variable_mapping_safe.csv` in target_recipes/
- 91 targets per state (vs 147 with full recipe)
- **All 52 state areas solved** (no PrimalInfeasible!) — 2.8 min with 8 workers
- CA: 91/91 targets hit (was PrimalInfeasible with full recipe)
- 25 states had some violated targets (small states mostly), but all solved

**Phase 14: SALT geographic distribution analysis** (2026-03-12)

Compared e18400 (SALT) geographic shares across three data sources using safe-recipe weights (which do NOT target SALT):

| Correlation pair | r |
|---|---|
| TMD (safe weights) vs Census (actual S&L tax collections) | **0.973** |
| TMD (safe weights) vs SOI (itemizer SALT deductions) | 0.882 |
| SOI vs Census | 0.783 |

Key insight: **TMD's SALT distribution from safe weights is much closer to Census actual tax collections than SOI is.** This is because:
- TMD e18400 = taxes *available* to deduct (all filers, no cap)
- SOI a18400 = taxes *actually deducted* (itemizers only, subject to $10K SALT cap)
- Census = actual S&L tax collections
- Post-TCJA, the $10K SALT cap and reduced itemization rates distort SOI's geographic distribution

Pattern: High-tax states (CA, NY, NJ, MD) are overstated in SOI shares (because they have more itemizers); no-income-tax states (TX, FL, WA, TN) are understated in SOI shares.

**Tax-Calculator SALT deduction simulation**:
- Used Tax-Calculator's computed c18300 (SALT after $10K cap) and c04470 (itemization indicator)
- TMD national SALT deducted: $131B vs SOI reported: $261B (TMD is about half of SOI — level mismatch)
- TMD-deducted vs SOI shares: r=0.864 (only modestly better than crude simulation)
- TMD-deducted vs TMD-available: r=0.999 (cap barely changes geographic distribution)
- Key: every PUF record with nonzero c18300 also itemizes (itemization bite = 0%)
- Note: TMD's PUF base starts from 2015, when there were many more itemizers (pre-TCJA). Available SALT for 2015 nonitemizers is not in the data — a bigger project.

**Census 2022 Census of Governments data** downloaded:
- Source: `https://www2.census.gov/programs-surveys/gov-finances/tables/2022/22slsstab1.xlsx`
- S&L general sales tax + property tax by state
- Saved to `/tmp/22slsstab1.xlsx` (needs permanent home if used for targets)
- Caveats: Census includes business-paid sales tax (not in individual TMD); fiscal year vs tax year mismatch; doesn't include nonfilers' taxes

### Strategy for SALT targets

The SOI SALT geographic distribution is distorted by TCJA (itemizer-only, $10K cap). We investigated multiple approaches:

**Phase 15: Census-share SALT targeting (e18400 available)**
- Used Census S&L tax collections (property + general sales) as share basis for e18400 targets
- `area_target = TMD_national_e18400 × Census_state_share`
- Result: r=1.000 for e18400 vs Census (perfect — targeted directly)
- But c18300 (deducted, post-cap) vs SOI a18300: r=0.918, CA still -7.2pp
- Problem: targeting "available" SALT can't fix the "deducted" distribution due to nonlinear $10K cap

**Phase 16: Hybrid Census/SOI shares for e18400**
- Blended Census and SOI shares: `hybrid = α×Census + (1-α)×SOI_a18300`
- Tested α = 0.50, 0.25, 0.00, 0.10 (4 runs, semi-binary search)
- Results: α=0.00 (100% SOI) was best but CA still -5.24pp
- Root cause: targeting the input variable (e18400, available) with output-concept shares (a18300, deducted) is a concept mismatch — the $10K cap changes geographic distribution nonlinearly

**Phase 17: Direct c18300 targeting — BREAKTHROUGH**
- Added c18300 (Tax-Calculator SALT after cap) and c04470 (itemized deductions) to solver data
  - Modified `_load_taxcalc_data()` in `create_area_weights_clarabel.py` to load from `cached_allvars.csv`
  - Fixed `_init_worker()` in `batch_weights.py` to use `_load_taxcalc_data()` instead of duplicating logic
- Targeted c18300 directly with SOI a18300 shares, AGI stubs 5-10 ($50K+)
- **Result: r=0.9999, mean|diff|=0.063pp, ALL states within 1pp!**
  - CA: -0.87pp (was -5.24pp with hybrid, -7.2pp with Census)
  - TX: -0.16pp, MD: -0.21pp, NY: -0.27pp
- Adding c18300 counts caused PrimalInfeasible for CA, NY, MA — amounts-only is the winner
- 97 targets per state (91 safe + 6 c18300 amount targets for stubs 5-10)
- NY had InsufficientProgress but still produced usable weights

**National level comparison (TMD Tax-Calculator outputs vs SOI)**:
- c18300 amount: TMD $131.1B vs SOI $120.3B (ratio 1.09, 9% high)
- c18300 count: TMD 16.48M vs SOI 14.70M (ratio 1.12)
- c04470 (total itemized): TMD $620.8B vs SOI $658.1B (ratio 0.94, 6% low)
- c04470 count: TMD 16.63M vs SOI 14.89M (ratio 1.12)

**Phase 18: Combined SALT targeting — e18400 + c18300 + e18500** (2026-03-12)

Tested targeting BOTH the input variable (e18400, SALT available, Census shares) and output variable (c18300, SALT deducted, SOI shares) simultaneously.

**Target recipe**: 91 safe + 6 c18300 (SOI a18300 shares) + 6 e18400 (Census combined S&L shares) + 6 e18500 (Census property shares), all AGI stubs 5-10 ($50K+) = **109 targets per state**.

**Result — ALL states solved, no failures!**

| Metric | c18300-only (97 tgts) | **Combined (109 tgts)** |
|---|---|---|
| c18300 vs SOI r | 0.9998 | **0.9998** (no degradation) |
| c18300 mean\|diff\| | 0.056pp | **0.054pp** |
| e18400 vs Census r | 0.9674 | **1.0000** |
| e18400 mean\|diff\| | 0.345pp | **0.009pp** |
| e18400 max\|diff\| (CA) | 3.902pp | **0.014pp** |
| e18500 vs Census r | 0.9535 | **0.9997** |
| e18500 mean\|diff\| | 0.419pp | **0.049pp** |

Key insight: the optimizer has enough degrees of freedom (~290K records × multipliers) to satisfy both the input-concept (Census SALT available) and output-concept (SOI SALT deducted) targets simultaneously. The earlier tension only arose when using wrong shares on a single variable.

**Timing comparison: Clarabel vs L-BFGS-B** (MN, 91 safe targets)

| Metric | Clarabel | L-BFGS-B |
|---|---|---|
| Wall clock time | **10.6s** | **69.3s** |
| Mean \|rel error\| | 0.376% | 0.007% |
| Max \|rel error\| | 0.400% | 0.112% |

**Clarabel is 6.5x faster.** Clarabel's errors cluster at the 0.4% tolerance boundary (by design — it minimizes weight distortion within the tolerance band, so constraints are binding). L-BFGS-B uses penalty optimization that drives errors closer to zero at the cost of 6x more time.

For 52 states at 8 workers: Clarabel batch ~186s (3.1 min) vs projected L-BFGS-B ~20 min.

### Current Best Recipe: 149 targets (CONFIRMED)

- **91 safe targets**: c00100 (AGI amounts + counts by filing status), e00200 (wages amt + nz count), e00300 (interest amt), e26270 (partnership/S-corp amt)
- **6 e18400 targets**: income/sales tax available, Census combined S&L shares, stubs 5-10
- **6 e18500 targets**: real estate tax available, Census property shares, stubs 5-10
- **6 c18300 targets**: SALT after $10K cap (Tax-Calculator output), SOI a18300 shares, stubs 5-10
- **6 e01700 targets**: taxable pensions, SOI a01700 shares, stubs 5-10
- **6 e01400 targets**: taxable IRA distributions, SOI a01400 shares, stubs 5-10
- **6 c02500 targets**: taxable SS (Tax-Calculator output), SOI a02500 shares, stubs 5-10
- **6 capgains_net targets**: net capital gains (p22250+p23250), SOI a01000 shares, stubs 5-10
- **6 e00600 targets**: ordinary dividends, SOI a00600 shares, stubs 5-10
- **6 e00900 targets**: business/professional income, SOI a00900 shares, stubs 5-10
- **1 eitc amount target**: aggregate EITC, SOI a59660 shares, all AGI
- **1 eitc count target**: EITC nonzero count, SOI n59660 shares, all AGI
- **1 ctc_total amount target**: aggregate CTC (a07225+a11070), SOI combined shares, all AGI
- **1 ctc_total count target**: CTC nonzero count, SOI combined n-shares, all AGI
- **Total: 149 targets per state** — all 51 states solve, 0 failures
- **Pipeline**: `python -m tmd.areas.prepare_and_solve --scope states --workers 8` (229s total)
- **Quality report**: `python -m tmd.areas.quality_report` → 25 states perfect, 26 with minor violations (85 total, 92% count targets)
- Census data: `tmd/areas/prepare/data/census_2022_state_local_finance.xlsx`
- Extended target config: `tmd/areas/prepare/extended_targets.py` (SOI_SHARED_SPECS + CENSUS_SHARED_SPECS + SOI_AGGREGATE_SPECS)

**Timing comparison (estimated):**
- Old L-BFGS-B method: ~60-90s/state sequential ≈ 50-90 min total
- New Clarabel + batch: ~4.5s/state effective (8 workers) ≈ **3.8 min total** (~15-20x speedup)

| Variable | Reference | r | mean|diff| | max|diff| (worst) |
|---|---|---|---|---|
| capgains_net (capital gains) | SOI A01000 | 0.9999 | 0.027pp | 0.183pp (FL) |
| c18300 (SALT deducted) | SOI A18300 | 0.9998 | 0.054pp | 0.353pp (CA) |
| e18400 (SALT available) | Census S&L | 1.0000 | 0.009pp | 0.064pp (NY) |
| e18500 (RE tax available) | Census property | 0.9997 | 0.049pp | 0.304pp (NY) |
| e01700 (taxable pension) | SOI A01700 | 0.9991 | 0.059pp | 0.739pp (CA) |
| e01400 (taxable IRA) | SOI A01400 | 0.9991 | 0.047pp | 0.530pp (CA) |
| c02500 (taxable SS) | SOI A02500 | 0.9994 | 0.040pp | 0.483pp (CA) |
| e00600 (dividends) | SOI A00600 | 0.999+ | ~0.05pp | ~0.5pp |
| e00900 (business income) | SOI A00900 | 0.999+ | ~0.05pp | ~0.5pp |
| eitc (EITC) | SOI A59660 | 1.0000 | 0.008pp | 0.043pp (CA) |
| ctc_total (CTC) | SOI A07225+A11070 | 1.0000 | 0.009pp | 0.079pp (CA) |

### Potential Next Steps

- **A. Full CD batch**: Run all 436 CDs with recipe; identify problem districts. Extend `extended_targets.py` for CDs.
- **B. Mortgage/charitable targeting**: Available vs deducted mismatch — investigate Tax-Calculator outputs.
- **D. Upstream prep**: Clean up for eventual PR — remove R dependency, ensure raw data in-repo, documentation.

**Phase 19: Pension and Social Security targeting** (2026-03-13)

**19A-B: Taxable targeting (e01700 + c02500)**
- Added `c02500` to `CACHED_TC_OUTPUTS` in `create_area_weights_clarabel.py` (joins c18300, c04470)
- `e01700` (taxable pensions) already in `tmd.csv.gz` — no loader change needed
- TMD vs SOI alignment: e01700 +0.0%, c02500 -0.5% (excellent)
- 121-target recipe (91 safe + 18 SALT + 6 e01700 + 6 c02500): all 51 states solve
- e01700 r=0.9991 vs SOI, c02500 r=0.9994 vs SOI

**19C: SSA total SS proxy**
- Downloaded SSA OASDI by State and County 2021 (`/tmp/oasdi_sc21.xlsx`, Table 3, monthly Dec 2021 benefits)
- SSA total OASDI ~$1,198B/yr vs TMD e02400 $799B (ratio 0.667 — TMD only has filers)
- SSA vs SOI taxable shares: r=0.9927 (high correlation — total and taxable SS have similar geographic distribution)

**19D: CPS total pension proxy**
- CPS ASEC 2022 (raw_cps_2022.h5): PNSN_VAL + ANN_VAL = $517B (only ~34% of TMD e01500 $1,508B)
- CPS misses IRA distributions, 401(k) withdrawals — conceptually much narrower than IRS "pensions and annuities"
- CPS pension state shares: r=0.9747 vs SOI, noisy (CA off by 2.76pp)
- EBRI: no downloadable state-level pension income data
- **Conclusion**: CPS pension is too noisy and conceptually mismatched to serve as a total pension proxy

**19E: SSA-based e02400 targeting — FAILED**
- Added 6 e02400 targets (SSA total shares, stubs 5-10) to 121-target recipe → 127 targets
- **Result: 50 of 51 states PrimalInfeasible** (only AZ solved)
- Root cause: SSA geographic distribution includes non-filers (~33% of SS recipients don't file tax returns). Non-filers are disproportionately concentrated in certain states. TMD only has filer records to reweight, so the SSA distribution is unachievable.
- This differs from SALT/Census: Census tax collections and SOI SALT both relate to the same filer/resident population, so their geographic patterns are compatible with filer-based reweighting.
- SSA targeting is unnecessary: c02500 targeting alone produces e02400 r=0.9928 vs SSA (untargeted total tracks well)

**19F: IRA distribution targeting (e01400) — SUCCESS**
- Key discovery: `e01400` (taxable IRA distributions) is separately identified in both TMD and SOI
  - TMD e01400: $406.3B vs SOI A01400: $407.5B (alignment: -0.3%)
  - IRS e01500 (total pensions $1,508B) = e01700 ($854B, taxable pensions) + e01400 ($406B, taxable IRA) + $249B nontaxable
  - IRA and pension geographic distributions differ meaningfully (r=0.984): FL has proportionally more IRA activity, CA/NY/VA have more employer pensions
- Added 6 e01400 targets (SOI A01400 shares, stubs 5-10) → 127-target recipe
- **Result: all 51 states solve, no degradation of any existing metric**

| Metric | 121 targets | 127 targets |
|---|---|---|
| e01400 vs SOI A01400 | r=0.9888, max 1.48pp | **r=0.9991, max 0.53pp** |
| Pension combined (e01400+e01700) | r=0.9973, max 1.06pp | **r=0.9992, max 0.71pp** |
| e01700, c02500, SALT metrics | unchanged | unchanged |

- Biggest IRA improvements: CA +1.48→+0.53pp, FL -1.09→-0.19pp, NY +1.13→+0.36pp
- 127 targets is the new best recipe

**Phase 19G: Capital gains targeting (capgains_net) — SUCCESS** (2026-03-13)
- SOI has `A01000` (net capital gain/loss); TMD has `p22250` (short-term) + `p23250` (long-term) both in `tmd.csv.gz`
- National alignment: TMD combined $1,891B vs SOI $2,050B (-7.8%) — acceptable for share-based targeting
- Added synthetic column `capgains_net = p22250 + p23250` to `_load_taxcalc_data()` in solver
- Added 6 `capgains_net` targets (SOI A01000 shares, stubs 5-10) → 133-target recipe
- **Result: all 51 states solve, capital gains r improved 0.9993→0.9999, zero degradation**

| Metric | 127 targets | 133 targets |
|---|---|---|
| capgains_net vs SOI A01000 | r=0.9993, max 0.618pp | **r=0.9999, max 0.183pp** |
| All other metrics | unchanged | unchanged |

- Biggest improvements: NY +0.516→-0.003pp, CA +0.618→+0.157pp, FL +0.588→+0.183pp

**Phase 19H: Dividends and business income targeting — COMPLETE** (2026-03-13)
- Ordinary dividends (`e00600`, SOI `A00600`): national alignment +0.7%, $387B
- Business/prof income (`e00900`, SOI `A00900`): national alignment -2.0%, $411B
- Both available in `tmd.csv.gz`, SOI A-variables available by state and AGI stub
- Quick test (CA + WY): both solve with 145 targets; CA all met, WY 1 violated
- Full 51-state run: **all solved, 0 failures, 224.4s with 8 workers**

**Phase 20A: Diagnostic report utility** (2026-03-13)
- Created `/tmp/area_diagnostic.py` — `state_diagnostic(st)` function produces per-state report
- Shows: weight distortion stats, all targeted variables (proportionate/target/achieved/% diff), important non-targeted variables vs SOI reference (including capital gains, dividends, business income, mortgage interest, charitable contributions, pensions, SS, itemized deductions, EITC, income tax)
- Tested on CA, VT, WY — all 127 targets within ±1% for all states

**Phase 19I: 2022 SOI shares test — COMPLETE** (2026-03-13)
- Ran 145-target recipe using 2022 SOI state shares (from `22in55cmcsv.csv`) with 2021 TMD national file
- **All 51 states solve, 0 failures, 267s with 8 workers**
- 31 states had minor violations (vs 26 with 2021 shares) — small increase as expected from larger share shifts
- Capital gains FL shift +4.25pp is the largest movement (2022 market downturn)
- Pipeline supports `--year 2022` for transparent switching
- Validates that recipe is robust to SOI year choice

**Phase 20C: Clean start-to-finish pipeline — COMPLETE** (2026-03-13)
- Created `tmd/areas/prepare/extended_targets.py` — formalizes the SOI-shared and Census-shared target logic that was previously in `/tmp` test scripts
  - `SOI_SHARED_SPECS`: c18300, e01700, c02500, e01400, capgains_net, e00600, e00900 (7 vars × 6 stubs = 42 targets)
  - `CENSUS_SHARED_SPECS`: e18400 (combined S&L), e18500 (property only) (2 vars × 6 stubs = 12 targets)
  - `append_extended_targets()`: reads existing base target CSVs, appends extended rows, writes back
  - Loads TMD data, SOI data by year, Census data automatically
- Updated `tmd/areas/prepare_and_solve.py`:
  - Now uses safe recipe (`states_safe.json`) as base (91 targets)
  - Calls `append_extended_targets()` to add 54 more (= 145 total)
  - Supports `--year` for SOI year selection (2021 default, 2022 available)
- **Full end-to-end test**: `python -m tmd.areas.prepare_and_solve --scope states --workers 8` → 51 states, 145 targets each, 0 failures, 230s total

**Phase 20E: Cross-state quality summary report — COMPLETE** (2026-03-13)
- Created `tmd/areas/quality_report.py` — parses all state solver logs, produces summary
  - Usage: `python -m tmd.areas.quality_report`
  - Shows: overall stats, target accuracy, weight distortion, per-state table, violations by variable, worst-5 violations
  - Per-state table has legend clarifying units (error = fraction, weight cols = multiplier on national weight)
- Key findings (51 states, 145 targets each):
  - 51/51 solved, 0 failures
  - 25 states hit all 145 targets; 26 states have minor violations (85 total)
  - Average hit rate: 98.9%, worst: 94.5% (DC)
  - Violations are 92% count targets (c00100 78/85, e00200 7/85) — zero amount violations
  - Weight RMSE: avg 0.415, worst WY 0.901. Best: PA 0.162, IL 0.186
  - Worst states (all small): DC 8, ND 7, WY 7, VT 7, SD 6, WV 6, AK 6

**TMD/PUF vs SOI geographic coverage:**
- PUF is a national sample with no state identifiers (FIPS=0 for all records)
- TMD PUF: 161.6M weighted returns, $14.845T AGI
- SOI US: 159.5M returns ($14.776T), = 50 states + DC (158.7M) + OA (0.72M, overseas/military) + PR (0.08M)
- OA+PR is only 0.5% of US total — minor issue for share-based targeting

**2021 vs 2022 SOI state share stability:**
- Most variables extremely stable: return counts r=0.9998, wages r=0.9992, IRA r=0.9999, pensions r=0.9999
- Dividends r=0.9943, partnership r=0.9869 — moderate shifts (FL gaining share)
- Capital gains r=0.9707 — FL share jumped 10.3%→14.6% (2022 market downturn hit states differently)
- Maximum shift: capital gains FL +4.25pp; most variables max shift <0.5pp

**National totals comparison (pension/SS variables):**

| Source | Pensions | IRA | Taxable SS | Total SS |
|---|---|---|---|---|
| TMD (PUF filers) | e01700: $854B | e01400: $406B | c02500: $412B | e02400: $799B |
| SOI | A01700: $854B | A01400: $408B | A02500: $414B | — |
| CPS (all persons) | PNSN+ANN: $517B | — | — | SS_VAL: $1,043B |
| SSA (all beneficiaries) | — | — | — | $1,198B |

## Branch

All work is on the `area-weighting-overhaul` branch. Push only to `origin` (donboyd5 fork).

## Files Created/Modified

New files:
- `tmd/areas/prepare/__init__.py`
- `tmd/areas/prepare/constants.py`
- `tmd/areas/prepare/census_population.py`
- `tmd/areas/prepare/soi_state_data.py`
- `tmd/areas/prepare/target_file_writer.py`
- `tmd/areas/prepare/target_sharing.py`
- `tmd/areas/prepare/soi_cd_data.py`
- `tmd/areas/prepare/cd_crosswalk.py`
- `tmd/areas/prepare/data/state_populations.json`
- `tmd/areas/prepare/data/cd_populations.json`
- `tmd/areas/prepare/data/geocorr2022_cd117_to_cd118.csv`
- `tmd/areas/create_area_weights_clarabel.py`
- `tmd/areas/targets/prepare/target_recipes/state_variable_mapping_allshares.csv`
- `tmd/areas/targets/prepare/target_recipes/cd_variable_mapping_allshares.csv`
- `tmd/areas/targets/prepare/target_recipes/states_safe.json` (minimal safe recipe)
- `tmd/areas/targets/prepare/target_recipes/state_variable_mapping_safe.csv` (safe variable mapping)
- `tmd/areas/batch_weights.py`
- `tmd/areas/prepare/extended_targets.py` (SOI-shared + Census-shared extended targets)
- `tmd/areas/prepare_and_solve.py`
- `tmd/areas/quality_report.py` (cross-state quality summary)
- `tmd/areas/targets/prepare/validation/` (old R-pipeline MN/MN01 reference files)

Modified files:
- `tmd/areas/create_area_weights.py` (added USE_CLARABEL bridge)
- `tmd/areas/prepare/constants.py` (added ALL_SHARING_MAPPINGS)
- `tmd/areas/prepare/target_sharing.py` (added all-shares pipeline + orchestrator + crosswalk integration)
- `tmd/areas/prepare/census_population.py` (moved data to JSON, added 2022)
- `tmd/areas/prepare/target_file_writer.py` (fixed allcount filter for shared names)
- `tmd/areas/prepare/cd_crosswalk.py` (fixed incorrect docstring re: 2022 boundaries)
- `tmd/areas/create_area_weights_clarabel.py` (load c18300, c04470, c02500 from cached_allvars; create capgains_net synthetic column)
- `tmd/areas/batch_weights.py` (use `_load_taxcalc_data()` in workers instead of duplicated loading)

## Open Items

- Both 2021 and 2022 CD SOI data are on 117th Congress boundaries. Crosswalk now integrated into pipeline (Phase 11) — `prepare_area_targets()` applies it by default for CDs.
- Decide on Clarabel multiplier bounds for area weights (currently [0.0, 100.0])
- When creating upstream PR: include only necessary source data (not spreadsheets, etc.)
- SALT targets: **RESOLVED** — combined targeting of e18400 (Census shares) + c18300 (SOI shares) + e18500 (Census shares) works perfectly.
- Pension/SS targets: **RESOLVED** — target taxable versions directly (e01700, e01400, c02500 with SOI shares).
- Capital gains: **RESOLVED** — synthetic `capgains_net = p22250 + p23250`, targeted with SOI A01000 shares.
- Dividends and business income: **RESOLVED** — e00600 with SOI A00600, e00900 with SOI A00900. All 51 states solve.
- Mortgage interest and charitable contributions: not yet targeted. Available vs deducted mismatch (e19200 $356B vs SOI a19300 $136B; e19800 $193B vs SOI a19700 $262B). Could target deducted amounts with SOI shares similar to SALT approach if Tax-Calculator outputs them separately.
- EITC and CTC credit targeting: **RESOLVED** — Phase 22 added aggregate amount + count targets per state. r=1.0000 for both, zero violations, zero degradation.
- c18300/c02500 count targets cause infeasibility — amounts-only for all Tax-Calculator output variables.
- Pipeline flexibility: confirmed that all targets recompute from TMD when tmd.csv.gz changes (share-based: TMD_national × SOI_share). Prerequisite: cached_allvars.csv and cached_c00100.npy must be regenerated if tmd.csv.gz changes.
- 2022 SOI state data: **TESTED** — all 51 states solve with 2022 shares. Pipeline supports `--year 2022`.
- Target-building logic: **FORMALIZED** — `tmd/areas/prepare/extended_targets.py` replaces `/tmp` test scripts.

**Phase 21: Credit targeting exploration (EITC + CTC)** (2026-03-13)

Investigated whether EITC and CTC can be targeted in area weighting. Focused on national-level alignment first.

**Credits covered**: EITC (Earned Income Tax Credit), CTC (Child Tax Credit) including nonrefundable and ACTC (Additional CTC, refundable). Dropped CDCC (too small at $3.5B). "Working Families Tax Credit" is a broad congressional concept covering EITC+CTC expansions.

**Law changes across years**:
- 2015 (pre-TCJA): CTC $1,000/child, partially refundable; EITC stable
- 2018-2020 (TCJA): CTC $2,000/child, phase-out $200K/$400K, ACTC up to $1,400-1,500
- 2021 (ARPA, one year only): CTC $3,000-3,600/child, fully refundable, no earned income req; EITC childless tripled
- 2022-2025 (TCJA): back to $2,000 CTC; ACTC up to $1,500-1,700
- 2026+ (OBBBA, signed July 2025): TCJA made permanent, CTC possibly ~$2,000-2,500
- Latest SOI data available: **2022** (SOI lags ~3 years)

**SOI state-level credit data availability**:
- EITC (a59660/n59660): all years 2015-2022, all AGI stubs — **available**
- CTC nonrefundable: a07220 (2015 only, pre-TCJA) → a07225 (2018+, post-TCJA, includes "other dependent credit")
- ACTC refundable (a11070/n11070): all years — **available**
- All credit variables already ingested by `soi_state_data.py` (generic CSV reader), just not wired as targets

**National alignment — PUF 2015 vs SOI 2015** (excellent):

| Credit | Item | PUF | SOI | Ratio |
|---|---|---|---|---|
| EITC | refundable amount | $59.0B | $58.5B | 1.01 |
| EITC | refundable count | 24.23M | 23.98M | 1.01 |
| CTC | nonrefundable amount | $27.2B | $26.9B | 1.01 |
| CTC | refundable (ACTC) amount | $26.7B | $26.1B | 1.02 |
| CTC | total amount | $53.9B | $53.1B | 1.01 |

**National alignment — TMD 2021 vs SOI 2021** (badly misaligned due to ARPA):

| Credit | Item | TMD | SOI | Ratio |
|---|---|---|---|---|
| EITC | total amount | $59.7B | $65.2B | 0.91 |
| EITC | total count | 15.45M | 32.07M | **0.48** |
| CTC | total amount | $249.2B | $122.9B | **2.03** |

TMD under ARPA law computes double the actual CTC and misses half the EITC claimants.

**National alignment — TMD 2022 vs SOI 2022** (rebuilt with TAXYEAR=2022, much better):

| Credit | Item | TMD | SOI | Ratio |
|---|---|---|---|---|
| EITC | total amount | $70.7B | $59.2B | 1.19 |
| EITC | total count | 33.62M | 23.69M | 1.42 |
| EITC | total average | $2,104 | $2,499 | 0.84 |
| CTC | nonrefundable amount | $92.7B | $82.9B | 1.12 |
| CTC | nonrefundable average | $2,207 | $2,177 | 1.01 |
| CTC | refundable (ACTC) amount | $36.3B | $33.9B | 1.07 |
| CTC | total amount | $129.0B | $116.7B | 1.11 |

Under 2022 TCJA law: CTC alignment greatly improved (1.11x vs 2.03x). EITC still 19% high on amount, 42% high on count.

**PUF vs CPS breakdown** (critical finding):
- Under **2022 TCJA**: 100% of all credits come from PUF records. CPS contributes $0. CPS records have median AGI=$0, no earned income.
- Under **2021 ARPA**: CPS contributes $20.0B CTC (8% of total) because ARPA CTC required no earned income. CPS EITC still $0.
- All 7,599 CPS records have `iitax` computed by Tax-Calculator. Under 2021, all get negative iitax (-$69.3B total) and refunds ($69.3B).

**Filer/nonfiler determination**:
- `is_tax_filer = (DATA_SOURCE == 1)` — defined in `tmd/utils/soi_replication.py` line 60
- This is used in **production** (imported by `reweight.py` at line 44, used at line 138)
- Perfect diagonal: PUF=filer, CPS=nonfiler. No income-based filer test applied post-construction.
- Tax-Calculator runs on ALL records regardless of filer status. CPS "nonfilers" get full tax calculations.
- Under policies that give refundable credits to zero-income households (like ARPA CTC), CPS records would receive credits — real people in that situation would file returns to claim refunds.

**National reweighting infrastructure**:
- `tmd/utils/reweight.py` targets 47 SOI variables — income, deductions, taxes. **No credit targets.**
- Only PUF records (filers) are constrained (line 138: `mask *= filer`)
- CPS records pass through unconstrained
- To target credits nationally: add credit targets to `tmd/storage/input/soi.csv`. Since all credits (under non-ARPA law) come from PUF records, the filer-only constraint works fine.

**Area weighting infrastructure for credits**:
- Solver already supports `scope=0` (all records), `scope=1` (PUF), `scope=2` (CPS) per target
- Current 144 of 145 targets use scope=1 (PUF only); 1 uses scope=0 (XTOT population)
- Credit targets would use scope=1, same as everything else
- Credit variables (`eitc`, `ctc_total`, `c11070`) are in `cached_allvars.csv` — just need to add to `CACHED_TC_OUTPUTS` in solver and add SOI share specs in `extended_targets.py`

**Assessment for area weighting**:
- EITC targeting (amounts, not counts) is promising: 2022 TMD amount 1.19x SOI, SOI has state-level data by AGI stub
- CTC targeting is feasible: 2022 TMD total 1.11x SOI (good), nonrefundable average essentially perfect (1.01x)
- Share-based targeting should work despite 10-20% level overstatement — same principle as c18300 (9% high nationally, r=0.9998 geographically)
- Use 2022 SOI shares to avoid ARPA distortions (even with 2021 TMD base)
- Don't target credit counts — count misalignment is worse than amount misalignment

**Open question**: Should credit targets be added to national reweighting first (so `tmd.csv.gz` starts with better credit totals), or only at area weighting stage? Adding nationally would benefit all downstream uses.

**Phase 22: Credit targeting at area level — EITC + CTC** (2026-03-14)

Added EITC and CTC (total) as aggregate state-level targets (amount + nonzero count per state, no AGI stub breakdown).

**Design decisions**:
- EITC is concentrated in AGI stubs 2-4 ($1-50K), zero above $50K — cannot use default stubs 5-10
- CTC spans stubs 3-8 but has large level misalignment (ARPA vs TCJA) in lower stubs
- Solution: target **aggregate amounts and counts** per state (one target each, all-AGI) rather than per-stub
- SOI shares: a59660 (EITC), a07225+a11070 (total CTC) — derived `a_ctc_total` in SOI loading
- Count targets: SOI n59660 (EITC), n07225+n11070 (total CTC) — derived `n_ctc_total`

**Implementation**:
- Added `eitc`, `ctc_total` to `CACHED_TC_OUTPUTS` in `create_area_weights_clarabel.py`
- Created `SOI_AGGREGATE_SPECS` in `extended_targets.py` — new spec type for aggregate (all-AGI) targets
- Added `_build_soi_aggregate_rows()` — builds amount (count=0) + nonzero count (count=2) rows with agilo=-9e99, agihi=9e99
- Created derived SOI variables `a_ctc_total` and `n_ctc_total` (= a/n07225 + a/n11070) in `_load_soi_by_stub()`
- Added `aggregate_specs` parameter to `append_extended_targets()`

**National alignment (TMD 2021 ARPA vs SOI 2021)**:
- EITC: TMD $59.7B vs SOI $65.2B (ratio 0.91) — 100% from PUF records
- CTC total: TMD $249.2B vs SOI $122.9B (ratio 2.03) — ARPA fully refundable vs TCJA

**Results — 149 targets, all 51 states solved, 0 failures, 229s**:

| Variable | Reference | r | mean|diff| | max|diff| (worst) |
|---|---|---|---|---|
| eitc (EITC amount) | SOI A59660 | 1.0000 | 0.008pp | 0.043pp (CA) |
| ctc_total (CTC amount) | SOI A07225+A11070 | 1.0000 | 0.009pp | 0.079pp (CA) |
| eitc count | SOI N59660 | (within ±0.40% tolerance) | — | — |
| ctc_total count | SOI N07225+N11070 | (within ±0.40% tolerance) | — | — |

- Zero credit target violations — both amounts and counts hit within tolerance for all 51 states
- Zero degradation of existing targets: 85 violations (same as 145-target baseline), all c00100/e00200 counts
- 25 states hit all 149 targets; 26 states have minor violations (same pattern as before)
- Despite 2x level misalignment on CTC (ARPA vs TCJA), share-based targeting works perfectly

## Resume Instructions

To continue this work in a new session, paste the following:

> Continue the area weighting system overhaul on the `area-weighting-overhaul` branch. Read the session notes at `session_notes/area_weighting_notes.md`. Phases 1-22 are complete. **149-target recipe fully confirmed** — all 51 states solve, 0 failures. Phase 22 added EITC+CTC credit targeting: aggregate amount + nonzero count per state (4 new targets), r=1.0000 for both credits vs SOI, zero degradation. Recipe: 91 safe + 42 SOI-shared by stub + 12 Census-shared by stub + 4 credit aggregate = 149 targets. Pending: (A) extend to CDs, (B) mortgage/charitable, (D) upstream prep. Key modules: `create_area_weights_clarabel.py` (solver), `extended_targets.py` (target specs), `batch_weights.py` (parallel runner), `quality_report.py`. Push only to `origin`, never upstream.
