"""
Target sharing — derive area targets from TMD national totals.

Replaces ``create_additional_state_targets.qmd``.

For variables where SOI doesn't directly match TMD definitions,
we use SOI data as the geographic distribution and TMD as levels:

    area_target = TMD_national_sum × (area_SOI / national_SOI)

Currently applied to 4 variables:
  - e01500 (pensions total) shared by SOI 01700 (taxable pensions)
  - e02400 (Social Security total) shared by SOI 02500 (taxable SS)
  - e18400 (SALT income/sales) shared by SOI 18400
  - e18500 (SALT real estate) shared by SOI 18500
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_tmd_national_sums(
    cached_allvars_path: Path,
    sharing_mappings: List[Tuple[str, str, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Compute TMD national weighted sums by AGI bin for shared vars.

    Reads cached_allvars.csv, filters to PUF records (data_source==1),
    and for each variable in sharing_mappings computes:
      - Nonzero count (weighted) per AGI bin
      - Dollar amount (weighted) per AGI bin
      - Totals across all bins (agistub=0)

    Parameters
    ----------
    cached_allvars_path : Path
        Path to ``cached_allvars.csv``.
    sharing_mappings : list of (tmdvar, soi_base, description)
        Variables to compute sums for.
    agi_cuts : list of float
        AGI bin cut points (same as used for SOI data).

    Returns
    -------
    pd.DataFrame
        Columns: tmdvar, basesoivname, agistub, scope, fstatus,
        count, tmdsum.
    """
    tmd = pd.read_csv(cached_allvars_path)
    # PUF-derived only
    tmd = tmd.loc[tmd["data_source"] == 1].copy()

    # Assign AGI stub
    tmd["agistub"] = (
        pd.cut(
            tmd["c00100"],
            bins=agi_cuts,
            right=False,
            labels=False,
        ).astype(int)
        + 1
    )  # 1-based

    # Compute sums per agistub
    records = []
    for tmdvar, soi_base, _ in sharing_mappings:
        for stub in sorted(tmd["agistub"].unique()):
            mask = tmd["agistub"] == stub
            vals = tmd.loc[mask, tmdvar]
            wts = tmd.loc[mask, "s006"]
            nzcount = (wts * (vals != 0)).sum()
            amount = (wts * vals).sum()
            records.append(
                {
                    "tmdvar": tmdvar,
                    "basesoivname": soi_base,
                    "agistub": stub,
                    "nzcount": nzcount,
                    "amount": amount,
                }
            )

    sums_by_stub = pd.DataFrame(records)

    # Add totals (agistub=0)
    totals = (
        sums_by_stub.groupby(["tmdvar", "basesoivname"])[["nzcount", "amount"]]
        .sum()
        .reset_index()
    )
    totals["agistub"] = 0
    sums_all = pd.concat([sums_by_stub, totals], ignore_index=True)

    # Pivot to long: nzcount and amount as separate rows
    sums_long = sums_all.melt(
        id_vars=["tmdvar", "basesoivname", "agistub"],
        value_vars=["nzcount", "amount"],
        var_name="vtype",
        value_name="tmdsum",
    )
    sums_long["fstatus"] = 0
    sums_long["scope"] = 1
    sums_long["count"] = np.where(sums_long["vtype"] == "nzcount", 2, 0)
    sums_long = sums_long.drop(columns=["vtype"])
    return sums_long


def compute_soi_shares(
    base_targets: pd.DataFrame,
    sharing_mappings: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Compute each area's share of the US total for sharer variables.

    For each (basesoivname, count, scope, fstatus, agistub):
      soi_share = area_target / US_target

    Parameters
    ----------
    base_targets : pd.DataFrame
        Base targets from SOI data.
    sharing_mappings : list
        Same as SHARING_MAPPINGS.

    Returns
    -------
    pd.DataFrame
        base_targets rows for sharer variables with added columns:
        soi_ussum, soi_share.
    """
    soi_bases = [m[1] for m in sharing_mappings]
    df = base_targets.loc[base_targets["basesoivname"].isin(soi_bases)].copy()
    # Get US total for each grouping
    group_cols = [
        "basesoivname",
        "count",
        "scope",
        "fstatus",
        "agistub",
    ]
    us_totals = df.loc[df["stabbr"] == "US", group_cols + ["target"]].rename(
        columns={"target": "soi_ussum"}
    )
    df = df.merge(us_totals, on=group_cols, how="left")
    df["soi_share"] = np.where(
        df["soi_ussum"] == 0, 0, df["target"] / df["soi_ussum"]
    )
    return df


def create_shared_targets(
    base_targets: pd.DataFrame,
    cached_allvars_path: Path,
    sharing_mappings: List[Tuple[str, str, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Create shared targets for all sharing variables.

    Combines SOI geographic shares with TMD national sums:
      area_target = tmdsum × soi_share

    For agistub=0 (totals), target = sum of bin targets.

    Returns DataFrame with same columns as base_targets plus tmdvar.
    """
    # Get TMD national sums
    tmd_sums = compute_tmd_national_sums(
        cached_allvars_path, sharing_mappings, agi_cuts
    )
    # Get SOI shares
    soi_shares = compute_soi_shares(base_targets, sharing_mappings)
    # Join shares with TMD sums
    joined = soi_shares.merge(
        tmd_sums[
            [
                "tmdvar",
                "basesoivname",
                "agistub",
                "scope",
                "fstatus",
                "count",
                "tmdsum",
            ]
        ],
        on=[
            "basesoivname",
            "scope",
            "fstatus",
            "count",
            "agistub",
        ],
        how="left",
    )

    # Calculate targets for non-total stubs
    joined["target"] = np.where(
        joined["agistub"] != 0,
        joined["tmdsum"] * joined["soi_share"],
        np.nan,
    )
    # Calculate totals (agistub=0) as sum of bin targets
    group_cols = [
        "stabbr",
        "tmdvar",
        "basesoivname",
        "scope",
        "fstatus",
        "count",
    ]
    bin_sums = (
        joined.loc[joined["agistub"] != 0]
        .groupby(group_cols)["target"]
        .sum()
        .reset_index()
        .rename(columns={"target": "target_total"})
    )
    joined = joined.merge(bin_sums, on=group_cols, how="left")
    joined.loc[joined["agistub"] == 0, "target"] = joined.loc[
        joined["agistub"] == 0, "target_total"
    ]
    joined = joined.drop(columns=["target_total"])

    # Construct new variable names
    # basesoivname → tmd{tmdvar_num}_shared_by_soi{soi_base}
    joined["basesoivname"] = (
        "tmd"
        + joined["tmdvar"].str[1:]
        + "_shared_by_soi"
        + joined["basesoivname"]
    )
    # soivname: prefix a/n based on count type
    joined["soivname"] = np.where(
        joined["count"] == 0,
        "a" + joined["basesoivname"],
        "n" + joined["basesoivname"],
    )

    # Select columns matching base_targets
    keep_cols = [
        "stabbr",
        "area",
        "count",
        "scope",
        "agilo",
        "agihi",
        "fstatus",
        "target",
        "basesoivname",
        "soivname",
        "agistub",
        "agilabel",
        "tmdvar",
    ]
    # Ensure area column exists
    if "area" not in joined.columns:
        joined["area"] = joined["stabbr"]
    result = joined[[c for c in keep_cols if c in joined.columns]]
    return result.copy()


def build_enhanced_targets(
    base_targets: pd.DataFrame,
    cached_allvars_path: Path,
    sharing_mappings: List[Tuple[str, str, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Combine base targets with shared targets into enhanced targets.

    Replaces ``combine_base_and_additional_targets.qmd``.

    Returns
    -------
    pd.DataFrame
        Enhanced targets with sort column, ready for target_file_writer.
    """
    shared = create_shared_targets(
        base_targets,
        cached_allvars_path,
        sharing_mappings,
        agi_cuts,
    )
    # Stack: base + shared (keep common columns)
    base_cols = [c for c in base_targets.columns if c != "tmdvar"]
    shared_subset = shared[[c for c in base_cols if c in shared.columns]]
    stack = pd.concat(
        [base_targets[base_cols], shared_subset], ignore_index=True
    )
    # Add sort: XTOT first, then everything else
    is_xtot = (
        (stack["basesoivname"] == "XTOT")
        & (stack["soivname"] == "XTOT")
        & (stack["scope"] == 0)
    )
    stack["_xtot"] = is_xtot.astype(int)
    stack = stack.sort_values(
        [
            "stabbr",
            "_xtot",
            "scope",
            "fstatus",
            "basesoivname",
            "count",
            "agistub",
        ],
        ascending=[True, False, True, True, True, True, True],
    ).reset_index(drop=True)
    stack["sort"] = stack.groupby("stabbr").cumcount() + 1
    stack = stack.drop(columns=["_xtot"])
    return stack
