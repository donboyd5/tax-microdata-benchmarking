**Read `repo_conventions_session_notes.md` first.**

# Session Notes: Congressional District Targets and Weights Pipeline
*Branch: `improve-potential-targets-structure`*
*Created: 2026-03-21*

## Background

Previously we improved the weighting process for national targets for the TMD file and overhauled the state area weights process completely. Related session notes: `area_weighting_notes.md` and `nlp_reweighting_session_notes.md`.

Now we are going to develop area targets and weights for congressional districts, for the 2022 tax year.

## Current State Pipeline (on master)

```bash
make clean
make data                                          # makes national TMD data; all non-skipped tests pass
python -m tmd.areas.prepare_targets --scope states  # create targets for 51 states/DC
python -m pytest tests/test_prepare_targets.py -v   # all target tests should pass
python -m tmd.areas.solve_weights --scope states --workers 8  # create the state weights
python -m pytest tests/test_solve_weights.py -v     # all weight tests should pass
python -m tmd.areas.quality_report                  # prints to the log
```

## Congressional District Challenges

Congressional district targets and weighting should be very similar to the state process. However, there are additional challenges that require data exploration before we begin:

- **SOI data structure differences:** CDs may have a different number of income ranges than states, and may have different variables. We need a complete inventory of these differences.
- **Coverage gaps:** We need to determine which congressional districts are present in the IRS data, how many might be missing, and what their population adds up to compared to the relevant national population.
- **District boundary crosswalk:** Both 2021 and 2022 SOI CD data use 117th Congress boundaries. We need a crosswalk to 118th Congress (current) boundaries using `geocorr2022_2607106953.csv`. Code that created a crosswalk from this file should exist somewhere in Git history.
- **Raw SOI CD data:** The raw congressional district SOI data should be somewhere in Git as well.

After the analysis, we may need to make minor adjustments to the proportion-of-national-population approach we used for states.

## Exploration Findings (2026-03-21)

### Data Structure
- Both 2021 and 2022 CD data have **9 AGI bins** (stubs 1–9), confirming `CD_AGI_CUTS` in constants.py.
- 2022 CD: 161 data columns; 2022 state: 161 data columns. Nearly identical — CD has `A00101` (state doesn't), state has `MVITA` (CD doesn't).
- 14 columns changed between 2021→2022 CD data (some dropped, some added — normal year-over-year SOI changes).

### Coverage
- **All 436 CDs are present** in both 2021 and 2022 data. The 8 at-large/single-CD states (AK, DC, DE, MT, ND, SD, VT, WY) use `CONG_DISTRICT=0` instead of `1` — recode to `1` in the pipeline.
- For multi-CD states, CD counts match 117th Congress expectations exactly.
- CDs sum exactly to the CD file's own US aggregate (ratio = 1.0000 for N1, AGI) once at-large states are included.

### CD File vs State SOI File
- The CD and state files are **different SOI products** with slightly different coverage.
- CD file state totals are ~98.3% of state file state totals for N1 (returns), ~98.3% for AGI on average.
- At-large states match almost exactly on returns (ratio ~1.0000) but can differ on AGI (e.g., MT = 0.9502).
- Multi-CD states range from 0.975 to 0.999.
- **Decision: Use CD file's own totals as denominators** for share computation, not state file totals. This keeps shares internally consistent and summing to 1.0.

### Crosswalk
- Geocorr crosswalk has 1,448 rows mapping 117th→118th Congress districts.
- Includes population-weighted allocation factors (`afact2` = cd117-to-cd118).
- Has a label/header row that needs skipping.
- Many CDs split across boundaries (e.g., NC-03 splits with factor 0.858).

### Comparison: States vs CDs vs Counties

| Attribute | States | CDs | Counties |
|-----------|--------|-----|----------|
| Count | 51 | 436 | 3,143 |
| AGI stubs | 10 | 9 | 8 |
| Total row in data | Yes (stub 0) | Yes (stub 0) | Separate file |
| Crosswalk needed | No | Yes (117th→118th) | No |
| Coverage | 100% | 100% (with at-large recode) | 100% |
| Variables | 161 data cols | 161 data cols | 161 data cols |
| Smallest area (returns) | WY ~281K | ~70K | Loving TX: 40 |

### Data Locations
- 2021 CD data: `tmd/areas/targets/prepare/prepare_cds/data/data_raw/congressional2021.zip` (contains `21incd.csv`)
- 2022 CD data: `tmd/areas/targets/prepare/prepare_cds/data/data_raw/22incd.csv`
- 2022 CD docguide: `tmd/areas/targets/prepare/prepare_cds/data/data_raw/22incddocguide.docx`
- Exploration script: `tmd/areas/explore_cd_data.py`

## County Analysis

See `county_weighting_notes.md` for detailed county feasibility analysis. County data is stored on a separate `county-data` branch pushed to origin fork, not merged into CD or master branches.
