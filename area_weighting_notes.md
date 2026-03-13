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

### Current Best Recipe: Safe + SALT + Pension/IRA + SS (127 targets)

- **91 safe targets**: c00100 (AGI amounts + counts by filing status), e00200 (wages amt + nz count), e00300 (interest amt), e26270 (partnership/S-corp amt)
- **6 e18400 targets**: income/sales tax available, Census combined S&L shares, stubs 5-10
- **6 e18500 targets**: real estate tax available, Census property shares, stubs 5-10
- **6 c18300 targets**: SALT after $10K cap (Tax-Calculator output), SOI a18300 shares, stubs 5-10
- **6 e01700 targets**: taxable pensions, SOI a01700 shares, stubs 5-10
- **6 e01400 targets**: taxable IRA distributions, SOI a01400 shares, stubs 5-10
- **6 c02500 targets**: taxable SS (Tax-Calculator output), SOI a02500 shares, stubs 5-10
- **Total: 127 targets per state**
- All 51 states solve; no degradation from 109-target recipe
- Census data: `tmd/areas/prepare/data/census_2022_state_local_finance.xlsx`
- SSA data (downloaded, for analysis): `/tmp/oasdi_sc21.xlsx`
- Target-building scripts: `/tmp/target_both_salt.py` (Phase 18), `/tmp/target_ira_test.py` (Phase 19F)

| Variable | Reference | r | mean|diff| | max|diff| (worst) |
|---|---|---|---|---|
| c18300 (SALT deducted) | SOI A18300 | 0.9998 | 0.054pp | 0.353pp (CA) |
| e18400 (SALT available) | Census S&L | 1.0000 | 0.009pp | 0.064pp (NY) |
| e18500 (RE tax available) | Census property | 0.9997 | 0.049pp | 0.304pp (NY) |
| e01700 (taxable pension) | SOI A01700 | 0.9991 | 0.059pp | 0.739pp (CA) |
| e01400 (taxable IRA) | SOI A01400 | 0.9991 | 0.047pp | 0.530pp (CA) |
| c02500 (taxable SS) | SOI A02500 | 0.9994 | 0.040pp | 0.483pp (CA) |

### Potential Next Steps

- **A. Full CD batch**: Run all 436 CDs with 127-target recipe; identify problem districts.
- **B. DC as state**: Treat DC as a state (10 AGI bins, state recipe) rather than CD.
- **C. Consider c04470 targets**: Total itemized deductions — TMD/SOI are within 6%, good candidate.
- **D. Upstream prep**: Clean up for eventual PR — remove R dependency, ensure raw data in-repo, documentation.
- **E. Formalize target builder**: Turn `/tmp/target_ira_test.py` into a proper pipeline module that the batch solver can call.
- **F. Explore total pension/SS proxies further**: CPS pension too noisy/narrow; SSA total SS caused PrimalInfeasible (non-filer mismatch). Could explore BEA REIS data or other sources.

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
- `tmd/areas/prepare_and_solve.py`
- `tmd/areas/targets/prepare/validation/` (old R-pipeline MN/MN01 reference files)

Modified files:
- `tmd/areas/create_area_weights.py` (added USE_CLARABEL bridge)
- `tmd/areas/prepare/constants.py` (added ALL_SHARING_MAPPINGS)
- `tmd/areas/prepare/target_sharing.py` (added all-shares pipeline + orchestrator + crosswalk integration)
- `tmd/areas/prepare/census_population.py` (moved data to JSON, added 2022)
- `tmd/areas/prepare/target_file_writer.py` (fixed allcount filter for shared names)
- `tmd/areas/prepare/cd_crosswalk.py` (fixed incorrect docstring re: 2022 boundaries)
- `tmd/areas/create_area_weights_clarabel.py` (load c18300, c04470, c02500 from cached_allvars for targeting)
- `tmd/areas/batch_weights.py` (use `_load_taxcalc_data()` in workers instead of duplicated loading)

## Open Items

- Both 2021 and 2022 CD SOI data are on 117th Congress boundaries. Crosswalk now integrated into pipeline (Phase 11) — `prepare_area_targets()` applies it by default for CDs.
- Decide on Clarabel multiplier bounds for area weights (currently [0.0, 100.0])
- When creating upstream PR: include only necessary source data (not spreadsheets, etc.)
- SALT targets: **RESOLVED** — combined targeting of e18400 (Census shares) + c18300 (SOI shares) + e18500 (Census shares) works perfectly.
- Pension/SS targets: **RESOLVED** — target taxable versions directly (e01700, e01400, c02500 with SOI shares). Total pension/SS proxies investigated:
  - SSA total SS: r=0.9927 vs SOI but caused PrimalInfeasible because SSA includes non-filers (unachievable distribution for filer-only reweighting)
  - CPS total pension: only 34% of IRS concept (misses IRA/401k distributions), noisy (r=0.975)
  - IRA distributions (e01400) are separately identified and well-aligned (-0.3%) — added to recipe
  - Untargeted e01500 (total pension) achieves r=0.9956 vs SOI A01700; e02400 (total SS) achieves r=0.9928 vs SSA — good enough from taxable targeting alone
  - Remaining nontaxable pension ($249B) has no proxy — only 16% of e01500, acceptable
- c18300/c02500 count targets cause infeasibility — amounts-only for all Tax-Calculator output variables
- Target-building logic in `/tmp/target_ira_test.py` — needs formalization into pipeline module

## Resume Instructions

To continue this work in a new session, paste the following:

> Continue the area weighting system overhaul on the `area-weighting-overhaul` branch. Read the session notes at `session_notes/area_weighting_notes.md`. Phases 1-19 are complete. Current best recipe: **127 targets** = 91 safe + 6 e18400 (Census S&L shares) + 6 e18500 (Census property shares) + 6 c18300 (SOI a18300 shares) + 6 e01700 (SOI a01700 shares) + 6 e01400 (SOI a01400 shares) + 6 c02500 (SOI a02500 shares), all stubs 5-10 ($50K+). All 51 states solve. Solver loads c18300, c02500 from cached_allvars.csv via `_load_taxcalc_data()`. Target-building script at `/tmp/target_ira_test.py` (needs formalization into pipeline module). Census data at `tmd/areas/prepare/data/census_2022_state_local_finance.xlsx`. SSA data at `/tmp/oasdi_sc21.xlsx` (for analysis, not targeting). Next steps: (A) extend to CDs, (B) consider c04470 targets, (C) formalize combined target builder, (D) upstream prep. Key files: `tmd/areas/create_area_weights_clarabel.py` (solver), `tmd/areas/batch_weights.py` (parallel runner), `tmd/areas/targets/prepare/target_recipes/states_safe.json` (safe recipe). Push only to `origin`, never upstream.
