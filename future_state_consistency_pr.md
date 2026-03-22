**Read `repo_conventions_session_notes.md` first.**

# Future State Consistency PR — Tracking List
*Created: 2026-03-22*

Changes to consider for a future PR that aligns the state pipeline
with the patterns established for CDs and (eventually) counties.

## Potential Changes

1. **Use N2 instead of Census population for state XTOT.**
   Currently states use Census population data (from `census_population.py`
   and `state_populations.json`). CDs use N2 from the SOI CD file.
   Using N2 for states too would:
   - Make the pipeline consistent across area types.
   - Remove dependency on the external Census population JSON file.
   - N2 (exemptions) is already in the SOI state CSV data.
   - Trade-off: Census population includes non-filers; N2 is filer-only.
     But since XTOT only sets proportions (pop_share), what matters is
     that shares are reasonable, not that they match actual population.

2. **Use `area` column consistently instead of `stabbr` for states.**
   The CD pipeline uses `area` as the geographic identifier throughout
   (e.g., "AL01"). The state pipeline uses `stabbr` in some places and
   `area` in others (area is set = stabbr late in the pipeline).
   Standardizing on `area` everywhere would simplify code that handles
   both area types and reduce the need for parallel functions
   (`_apply_sharing` vs `_apply_cd_sharing`, `build_all_shares_targets`
   vs `build_cd_shares_targets`).

3. **Unify share computation functions.**
   `compute_soi_shares` (states) and `compute_cd_soi_shares` (CDs) do
   essentially the same thing but differ in how they find the national
   total (US row vs sum-of-areas) and rescaling (states rescale to 1.0,
   CDs don't need it). Could be unified with a parameter.

4. **Unify `_apply_sharing` and `_apply_cd_sharing`.**
   These differ only in whether they group by `stabbr` or `area`. If
   states used `area` consistently, one function would suffice.

5. **Unify `build_all_shares_targets` and `build_cd_shares_targets`.**
   Same as above — these are parallel due to the stabbr/area split.

## County-Specific Notes

When adding counties, watch for:

- **8 AGI stubs** (fewer than CDs' 9 or states' 10). Need `COUNTY_AGI_CUTS`.
- **Separate total file.** County data has with-AGI and no-AGI CSVs that
  must be merged. The total row (stub 0) is in a different file.
- **Encoding:** County CSV uses latin-1 (Dona Ana county), not UTF-8.
- **FIPS codes:** Need consistent 5-digit zero-padding (2-digit state +
  3-digit county). Some "counties" are independent cities (VA), parishes
  (LA), boroughs (AK).
- **No crosswalk needed.** County boundaries are stable (unlike CDs).
- **Solver feasibility for small counties.** 115 counties have <1K returns.
  May need tiered recipes (fewer targets, merged AGI bins, wider tolerances).
- **Data suppression.** Small counties have SOI-suppressed cells (zeros that
  mean "confidential" not "truly zero"). Need to distinguish.
- **Scale:** 3,143 areas. ~1.5 hours at current solve speed (16 workers).
  Optimization C (relaxed tolerances) would help.
- **Recipe design:** One-size-fits-all won't work. Tiered approach needed
  (full recipe for large counties, minimal for tiny ones).
