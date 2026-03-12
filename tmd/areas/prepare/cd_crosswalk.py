"""
Congressional district crosswalk: 117th → 118th session boundaries.

Both 2021 and 2022 SOI CD data use 117th Congress boundaries
(2010 Census), as confirmed by IRS documentation and by checking
the actual CD counts per state (e.g., OR=5 not 6, TX=36 not 38).

The 118th Congress boundaries (2020 Census) differ significantly
in some states. This module allocates 117th targets to 118th
districts using population-weighted shares from a geocorr crosswalk.

The crosswalk is needed for BOTH 2021 and 2022 SOI CD data.

Crosswalk source: Missouri Census Data Center, Geocorr 2022.
"""

from pathlib import Path

import pandas as pd


def load_geocorr_crosswalk(crosswalk_path: Path) -> pd.DataFrame:
    """
    Load and clean a Geocorr 2022 crosswalk CSV.

    Applies the same cleaning as the R pipeline:
      - Skip label row (row 1 after header).
      - Pad NC cd117 codes to 2 digits.
      - Remap DC98 → DC00.
      - Remove PR records.
      - Remove cd117=="-" records.
      - Compute population-weighted share117to118 from pop2020.

    Parameters
    ----------
    crosswalk_path : Path
        Path to the geocorr CSV (e.g., geocorr2022_2428906586.csv).

    Returns
    -------
    pd.DataFrame
        Columns: stabbr, cd117, cd118, statecd117, statecd118,
        share117to118.
    """
    raw = pd.read_csv(crosswalk_path)
    # First row is labels — skip it
    raw = raw.iloc[1:].copy()
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"stab": "stabbr", "pop20": "pop2020"})
    # Convert numeric columns
    for col in ["pop2020", "afact", "afact2"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Remove PR and cd117=="-"
    raw = raw.loc[raw["stabbr"] != "PR"].copy()
    raw = raw.loc[raw["cd117"] != "-"].copy()

    # Fix NC: pad cd117 to 2 digits
    nc_mask = (raw["stabbr"] == "NC") & (raw["cd117"].str.len() != 2)
    raw.loc[nc_mask, "cd117"] = (
        raw.loc[nc_mask, "cd117"].astype(int).astype(str).str.zfill(2)
    )

    # Build area codes
    raw["statecd117"] = raw["stabbr"] + raw["cd117"]
    raw["statecd118"] = raw["stabbr"] + raw["cd118"]

    # Remap DC98 → DC00
    raw["statecd117"] = raw["statecd117"].replace("DC98", "DC00")
    raw["statecd118"] = raw["statecd118"].replace("DC98", "DC00")

    # Compute population-weighted share (more precise than afact)
    raw["share117to118"] = raw.groupby("statecd117")["pop2020"].transform(
        lambda x: x / x.sum()
    )
    raw["share117to118"] = raw["share117to118"].fillna(0)

    result = raw[
        [
            "stabbr",
            "cd117",
            "cd118",
            "statecd117",
            "statecd118",
            "share117to118",
        ]
    ].copy()
    return result.reset_index(drop=True)


def allocate_117_to_118(
    targets_117: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Allocate 117th session CD targets to 118th session boundaries.

    For each 118th CD, sums:
      target_118 = sum(target_117 × share117to118)
    across all contributing 117th CDs.

    Parameters
    ----------
    targets_117 : pd.DataFrame
        Enhanced targets for 117th session CDs. Must have 'area'
        column matching statecd117 codes (e.g., "NY01").
    crosswalk : pd.DataFrame
        Output of ``load_geocorr_crosswalk()``.

    Returns
    -------
    pd.DataFrame
        Targets allocated to 118th session CD boundaries.
    """
    # Join crosswalk to 117th targets
    joined = crosswalk.rename(
        columns={"statecd118": "area118", "statecd117": "area117"}
    ).merge(
        targets_117.rename(columns={"area": "area117"}),
        on=["stabbr", "area117"],
        how="left",
        # Many crosswalk rows can match many target rows
    )

    # Allocate targets
    joined["target"] = joined["target"] * joined["share117to118"]

    # Aggregate to 118th CD level
    group_cols = [
        "stabbr",
        "area118",
        "basesoivname",
        "soivname",
        "scope",
        "fstatus",
        "count",
        "agistub",
        "agilo",
        "agihi",
        "agilabel",
    ]
    # Only keep group cols that exist in joined
    group_cols = [c for c in group_cols if c in joined.columns]

    result = (
        joined.groupby(group_cols, as_index=False)["target"]
        .sum()
        .rename(columns={"area118": "area"})
    )
    return result


def combine_sessions(
    targets_117: pd.DataFrame,
    targets_118: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stack 117th and 118th session targets with a session column.

    Parameters
    ----------
    targets_117 : pd.DataFrame
        Enhanced targets for 117th session.
    targets_118 : pd.DataFrame
        Enhanced targets for 118th session.

    Returns
    -------
    pd.DataFrame
        Stacked DataFrame with ``session`` column (117 or 118).
    """
    t117 = targets_117.copy()
    t118 = targets_118.copy()
    t117["session"] = 117
    t118["session"] = 118
    # Use only common columns
    common = sorted(set(t117.columns) & set(t118.columns))
    return pd.concat([t117[common], t118[common]], ignore_index=True)
