"""
Extended targets — append SOI-shared and Census-shared targets to area files.

After the base pipeline writes per-area target files (using the recipe + sharing
system), this module appends additional targets that use different geographic
distribution sources:

  1. **SOI-shared**: variable's TMD national sum × (state SOI / US SOI)
     for variables where TMD and SOI definitions match well enough to use
     SOI geographic shares directly.

  2. **Census-shared**: variable's TMD national sum × Census tax share,
     distributed across AGI bins by SOI bin proportions.  Used for SALT
     variables where Census provides better geographic distribution than SOI.

All targets are restricted to high-income AGI stubs (default: $50K+,
stubs 5-10) to avoid noisy low-income bins.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tmd.areas.prepare.constants import (
    ALL_STATES,
    STATE_AGI_CUTS,
    AreaType,
    SOI_STATE_CSV_PATTERNS,
    build_agi_labels,
)

# --- Extended target configuration ---

# SOI-shared targets: (tmd_varname, soi_amount_varname)
# Each gets one target per AGI stub in the stub list.
SOI_SHARED_SPECS: List[Tuple[str, str]] = [
    ("c18300", "a18300"),  # Total SALT deduction (Tax-Calc output)
    ("e01700", "a01700"),  # Taxable pensions
    ("c02500", "a02500"),  # Taxable Social Security
    ("e01400", "a01400"),  # Taxable IRA distributions
    ("capgains_net", "a01000"),  # Net capital gains (p22250+p23250)
    ("e00600", "a00600"),  # Ordinary dividends
    ("e00900", "a00900"),  # Business/professional net income
]

# Census-shared targets: (tmd_varname, census_type)
# census_type determines which Census tax measure to use for state shares.
CENSUS_SHARED_SPECS: List[Tuple[str, str]] = [
    ("e18400", "combined"),  # SALT income/sales → Census property+sales
    ("e18500", "property"),  # SALT real estate → Census property only
]

# Default AGI stubs for extended targets ($50K+)
DEFAULT_EXTENDED_STUBS = [5, 6, 7, 8, 9, 10]

# Name-to-abbreviation mapping for Census data
_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

# --- Census data loading ---

_CENSUS_DATA_PATH = (
    Path(__file__).parent / "data" / "census_2022_state_local_finance.xlsx"
)


def _load_census_shares(
    census_path: Path = _CENSUS_DATA_PATH,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load Census state/local finance data and compute state tax shares.

    Returns
    -------
    (combined_shares, property_shares) : tuple of dicts
        combined_shares: state → (property + sales) / US total
        property_shares: state → property / US property total
    """
    cdf = pd.read_excel(census_path, sheet_name="2022_US_WY", header=None)
    sl_cols = {}
    seen = set()
    for col_idx in range(2, len(cdf.columns)):
        n = (
            str(cdf.iloc[8, col_idx]).strip()
            if pd.notna(cdf.iloc[8, col_idx])
            else ""
        )
        gt = (
            str(cdf.iloc[9, col_idx]).strip()
            if pd.notna(cdf.iloc[9, col_idx])
            else ""
        )
        at = (
            str(cdf.iloc[11, col_idx]).strip()
            if pd.notna(cdf.iloc[11, col_idx])
            else ""
        )
        if n and gt == "State & local" and at == "amount1" and n not in seen:
            seen.add(n)
            sl_cols[n] = col_idx

    us_col = sl_cols["United States Total"]
    us_property = float(cdf.iloc[25, us_col])
    us_sales = float(cdf.iloc[27, us_col])

    combined_shares = {}
    property_shares = {}
    for name, col in sl_cols.items():
        abbr = _NAME_TO_ABBR.get(name)
        if abbr is None:
            continue
        pt = float(cdf.iloc[25, col]) if pd.notna(cdf.iloc[25, col]) else 0
        gs = float(cdf.iloc[27, col]) if pd.notna(cdf.iloc[27, col]) else 0
        combined_shares[abbr] = (pt + gs) / (us_property + us_sales)
        property_shares[abbr] = pt / us_property

    return combined_shares, property_shares


# --- SOI data loading ---


def _load_soi_by_stub(soi_year: int) -> pd.DataFrame:
    """
    Load raw SOI state CSV for the given year.

    Returns DataFrame with lowercase columns, filtered to agi_stub > 0,
    with a 'stabbr' column (uppercase state abbreviation).
    """
    soi_raw_dir = (
        Path(__file__).parent.parent
        / "targets"
        / "prepare"
        / "prepare_states"
        / "data"
        / "data_raw"
    )
    fname = SOI_STATE_CSV_PATTERNS.get(soi_year)
    if fname is None:
        raise ValueError(
            f"No SOI state CSV for year {soi_year}. "
            f"Available: {sorted(SOI_STATE_CSV_PATTERNS.keys())}"
        )
    soi = pd.read_csv(soi_raw_dir / fname, thousands=",")
    soi.columns = [c.lower() for c in soi.columns]
    soi_by_stub = soi[soi["agi_stub"] > 0].copy()
    soi_by_stub["stabbr"] = soi_by_stub["state"].str.strip().str.upper()
    return soi_by_stub


# --- TMD national sums ---


def _compute_tmd_stub_sums(
    vdf: pd.DataFrame,
    target_vars: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Compute PUF-weighted national sums by AGI stub for each variable.

    Returns dict: varname → {stub: weighted_sum}.
    """
    puf_mask = vdf["data_source"] == 1
    s006 = vdf["s006"]
    result = {}
    for var in target_vars:
        result[var] = {}
        for stub in range(1, 11):
            mask = puf_mask & (vdf["agistub"] == stub)
            result[var][stub] = float((s006[mask] * vdf.loc[mask, var]).sum())
    return result


# --- Target row builders ---


def _build_soi_stub_rows(
    st: str,
    varname: str,
    soi_varname: str,
    tmd_stub_sums: Dict[int, float],
    soi_by_stub: pd.DataFrame,
    agi_labels: pd.DataFrame,
    stubs: List[int],
) -> List[dict]:
    """Build target rows for one SOI-shared variable and one state."""
    rows = []
    st_soi = soi_by_stub[soi_by_stub["stabbr"] == st]
    us_soi = soi_by_stub[soi_by_stub["stabbr"] == "US"]
    for stub in stubs:
        sr = st_soi[st_soi["agi_stub"] == stub]
        usr = us_soi[us_soi["agi_stub"] == stub]
        if sr.empty or usr.empty:
            continue
        us_val = float(usr[soi_varname].values[0])
        if us_val <= 0:
            continue
        share = float(sr[soi_varname].values[0]) / us_val
        target = tmd_stub_sums[stub] * share
        agi_row = agi_labels[agi_labels["agistub"] == stub].iloc[0]
        rows.append(
            {
                "varname": varname,
                "count": 0,
                "scope": 1,
                "agilo": agi_row["agilo"],
                "agihi": agi_row["agihi"],
                "fstatus": 0,
                "target": target,
            }
        )
    return rows


def _build_census_stub_rows(
    st: str,
    varname: str,
    census_share: float,
    enhanced_rows: pd.DataFrame,
    tmd_stub_sums: Dict[int, float],
    agi_labels: pd.DataFrame,
    stubs: List[int],
) -> List[dict]:
    """Build target rows for one Census-shared variable and one state."""
    if census_share <= 0:
        return []
    st_rows = enhanced_rows[enhanced_rows["area"] == st]
    bin_rows = st_rows[st_rows["agistub"].isin(stubs)]
    if bin_rows.empty:
        return []
    soi_bt = bin_rows.set_index("agistub")["target"]
    soi_total = soi_bt.sum()
    if soi_total <= 0:
        return []
    bin_props = soi_bt / soi_total
    tmd_total = sum(tmd_stub_sums[s] for s in stubs)
    st_total = tmd_total * census_share

    rows = []
    for stub, prop in bin_props.items():
        agi_row = agi_labels[agi_labels["agistub"] == stub].iloc[0]
        rows.append(
            {
                "varname": varname,
                "count": 0,
                "scope": 1,
                "agilo": agi_row["agilo"],
                "agihi": agi_row["agihi"],
                "fstatus": 0,
                "target": st_total * prop,
            }
        )
    return rows


# --- Main entry point ---


def append_extended_targets(
    target_dir: Path,
    enhanced_targets: pd.DataFrame,
    soi_year: int = 2021,
    stubs: Optional[List[int]] = None,
    areas: Optional[List[str]] = None,
    soi_specs: Optional[List[Tuple[str, str]]] = None,
    census_specs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, int]:
    """
    Append extended (SOI-shared and Census-shared) targets to area files.

    This should be called after ``write_area_target_files()`` has written
    the base target CSVs.  It reads each area's file, appends new target
    rows, and writes back.

    Parameters
    ----------
    target_dir : Path
        Directory containing ``{area}_targets.csv`` files.
    enhanced_targets : pd.DataFrame
        The enhanced targets DataFrame from ``prepare_area_targets()``.
        Used for Census-shared bin proportioning (SALT rows).
    soi_year : int
        SOI data year for geographic shares (2021 or 2022).
    stubs : list of int, optional
        AGI stubs to target (default: 5-10, i.e., $50K+).
    areas : list of str, optional
        Area codes to process (default: ALL_STATES).
    soi_specs : list of (varname, soi_varname), optional
        SOI-shared target specs (default: SOI_SHARED_SPECS).
    census_specs : list of (varname, census_type), optional
        Census-shared target specs (default: CENSUS_SHARED_SPECS).

    Returns
    -------
    dict
        Mapping of area code → final target count.
    """
    from tmd.areas.create_area_weights_clarabel import _load_taxcalc_data

    if stubs is None:
        stubs = DEFAULT_EXTENDED_STUBS
    if areas is None:
        areas = ALL_STATES
    if soi_specs is None:
        soi_specs = SOI_SHARED_SPECS
    if census_specs is None:
        census_specs = CENSUS_SHARED_SPECS

    agi_labels = build_agi_labels(AreaType.STATE)

    # Load TMD data
    vdf = _load_taxcalc_data()
    # Load additional variables from cached_allvars
    repo_root = Path(__file__).parent.parent.parent
    allvars_path = repo_root / "tmd" / "storage" / "output" / "cached_allvars.csv"
    if allvars_path.exists():
        allvars = pd.read_csv(allvars_path)
        needed = {v for v, _ in soi_specs} | {v for v, _ in census_specs}
        for var in needed:
            if var not in vdf.columns and var in allvars.columns:
                vdf[var] = allvars[var].values

    # Assign AGI stubs
    vdf["agistub"] = (
        pd.cut(vdf["c00100"], bins=STATE_AGI_CUTS, right=False, labels=False)
        .astype(int)
        + 1
    )

    # Compute TMD national sums by stub
    all_vars = [v for v, _ in soi_specs] + [v for v, _ in census_specs]
    tmd_by_stub = _compute_tmd_stub_sums(vdf, all_vars)

    # Load SOI data
    soi_by_stub = _load_soi_by_stub(soi_year)

    # Load Census shares
    combined_shares, property_shares = _load_census_shares()

    # Prepare enhanced target subsets for Census-shared bin proportioning
    salt_rows = {}
    for varname, census_type in census_specs:
        # Match by basesoivname containing the 5-digit code
        code = varname[1:]  # e18400 → 18400
        salt_rows[varname] = enhanced_targets[
            enhanced_targets["basesoivname"].str.contains(code, na=False)
            & (enhanced_targets["count"] == 0)
        ].copy()

    # Process each area
    result = {}
    for st in areas:
        fpath = target_dir / f"{st.lower()}_targets.csv"
        if not fpath.exists():
            continue
        existing = pd.read_csv(fpath)
        new_rows = []

        # SOI-shared targets
        for varname, soi_varname in soi_specs:
            new_rows.extend(
                _build_soi_stub_rows(
                    st,
                    varname,
                    soi_varname,
                    tmd_by_stub[varname],
                    soi_by_stub,
                    agi_labels,
                    stubs,
                )
            )

        # Census-shared targets
        for varname, census_type in census_specs:
            if census_type == "combined":
                share = combined_shares.get(st, 0)
            else:
                share = property_shares.get(st, 0)
            new_rows.extend(
                _build_census_stub_rows(
                    st,
                    varname,
                    share,
                    salt_rows[varname],
                    tmd_by_stub[varname],
                    agi_labels,
                    stubs,
                )
            )

        if new_rows:
            existing = pd.concat(
                [existing, pd.DataFrame(new_rows)], ignore_index=True
            )

        existing.to_csv(fpath, index=False)
        result[st] = len(existing)

    return result
