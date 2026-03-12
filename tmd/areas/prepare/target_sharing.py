"""
Target sharing — derive area targets from TMD national totals.

Two modes:

1. **Legacy (4 vars)**: For variables where SOI doesn't directly match
   TMD definitions, use SOI as geographic distribution and TMD as levels.
   Called via ``build_enhanced_targets()``.

2. **All-shares**: Every targeted variable uses TMD national totals
   scaled by SOI geographic shares. Ensures area targets sum exactly
   to national TMD totals. Called via ``build_all_shares_targets()``.

Formula:
    area_target = TMD_national_sum × (area_SOI / national_SOI)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---- TMD national sums (legacy 3-tuple mappings) --------


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


# ---- TMD national sums (all-shares 5-tuple mappings) ----


def compute_tmd_national_sums_all(
    cached_allvars_path: Path,
    all_mappings: List[Tuple[str, str, int, int, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Compute TMD national sums for ALL targeted variable combos.

    Handles three count types:
      - count=0 (amounts): sum(s006 * var) per AGI bin
      - count=1 (allcounts): weighted count, optionally by MARS
      - count=2 (nonzero counts): sum(s006 * (var != 0))

    Parameters
    ----------
    cached_allvars_path : Path
        Path to ``cached_allvars.csv``.
    all_mappings : list of (tmdvar, soi_base, count, fstatus, desc)
        All variable/count/fstatus combinations.
    agi_cuts : list of float
        AGI bin cut points.

    Returns
    -------
    pd.DataFrame
        Columns: tmdvar, basesoivname, count, fstatus, agistub,
        scope, tmdsum.
    """
    tmd = pd.read_csv(cached_allvars_path)
    tmd = tmd.loc[tmd["data_source"] == 1].copy()

    tmd["agistub"] = (
        pd.cut(
            tmd["c00100"],
            bins=agi_cuts,
            right=False,
            labels=False,
        ).astype(int)
        + 1
    )

    records = []
    stubs = sorted(tmd["agistub"].unique())

    for tmdvar, soi_base, count_type, fstatus, _ in all_mappings:
        for stub in stubs:
            mask = tmd["agistub"] == stub
            wts = tmd.loc[mask, "s006"]

            if count_type == 0:
                # Amount: weighted sum
                vals = tmd.loc[mask, tmdvar]
                tmdsum = (wts * vals).sum()
            elif count_type == 1:
                # Allcount: weighted count of returns
                if fstatus == 0:
                    tmdsum = wts.sum()
                else:
                    mars_mask = tmd.loc[mask, "MARS"] == fstatus
                    tmdsum = wts[mars_mask].sum()
            elif count_type == 2:
                # Nonzero count
                vals = tmd.loc[mask, tmdvar]
                tmdsum = (wts * (vals != 0)).sum()
            else:
                tmdsum = 0.0

            records.append(
                {
                    "tmdvar": tmdvar,
                    "basesoivname": soi_base,
                    "count": count_type,
                    "fstatus": fstatus,
                    "agistub": stub,
                    "tmdsum": tmdsum,
                }
            )

    sums_by_stub = pd.DataFrame(records)

    # Add totals (agistub=0)
    group = ["tmdvar", "basesoivname", "count", "fstatus"]
    totals = sums_by_stub.groupby(group)[["tmdsum"]].sum().reset_index()
    totals["agistub"] = 0
    sums_all = pd.concat([sums_by_stub, totals], ignore_index=True)
    sums_all["scope"] = 1
    return sums_all


# ---- SOI geographic shares ----------------------------


def compute_soi_shares(
    base_targets: pd.DataFrame,
    sharing_mappings: List[Tuple],
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
        Either 3-tuple or 5-tuple mappings. Element [1] is soi_base.

    Returns
    -------
    pd.DataFrame
        base_targets rows for sharer variables with added columns:
        soi_ussum, soi_share.
    """
    soi_bases = [m[1] for m in sharing_mappings]
    df = base_targets.loc[base_targets["basesoivname"].isin(soi_bases)].copy()
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


# ---- Legacy shared targets (4 variables) ---------------


def create_shared_targets(
    base_targets: pd.DataFrame,
    cached_allvars_path: Path,
    sharing_mappings: List[Tuple[str, str, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Create shared targets for legacy sharing variables.

    Combines SOI geographic shares with TMD national sums:
      area_target = tmdsum × soi_share

    For agistub=0 (totals), target = sum of bin targets.

    Returns DataFrame with same columns as base_targets plus tmdvar.
    """
    tmd_sums = compute_tmd_national_sums(
        cached_allvars_path, sharing_mappings, agi_cuts
    )
    soi_shares = compute_soi_shares(base_targets, sharing_mappings)
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

    joined["target"] = np.where(
        joined["agistub"] != 0,
        joined["tmdsum"] * joined["soi_share"],
        np.nan,
    )
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

    joined["basesoivname"] = (
        "tmd"
        + joined["tmdvar"].str[1:]
        + "_shared_by_soi"
        + joined["basesoivname"]
    )
    joined["soivname"] = np.where(
        joined["count"] == 0,
        "a" + joined["basesoivname"],
        "n" + joined["basesoivname"],
    )

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

    Legacy approach: base SOI targets + 4 shared variables stacked.

    Returns
    -------
    pd.DataFrame
        Enhanced targets with sort column.
    """
    shared = create_shared_targets(
        base_targets,
        cached_allvars_path,
        sharing_mappings,
        agi_cuts,
    )
    base_cols = [c for c in base_targets.columns if c != "tmdvar"]
    shared_subset = shared[[c for c in base_cols if c in shared.columns]]
    stack = pd.concat(
        [base_targets[base_cols], shared_subset], ignore_index=True
    )
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


# ---- All-shares targets --------------------------------


def _apply_sharing(
    soi_shares: pd.DataFrame,
    tmd_sums: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join SOI shares with TMD sums and compute targets.

    For non-total stubs: target = tmdsum × soi_share
    For agistub=0: target = sum of bin targets
    """
    join_keys = [
        "basesoivname",
        "scope",
        "fstatus",
        "count",
        "agistub",
    ]
    joined = soi_shares.merge(
        tmd_sums[join_keys + ["tmdvar", "tmdsum"]],
        on=join_keys,
        how="left",
    )

    # Compute bin-level targets
    joined["target"] = np.where(
        joined["agistub"] != 0,
        joined["tmdsum"] * joined["soi_share"],
        np.nan,
    )

    # Compute totals (agistub=0) as sum of bin targets
    group_cols = [
        "stabbr",
        "tmdvar",
        "basesoivname",
        "scope",
        "fstatus",
        "count",
    ]
    # Keep only group cols that exist
    group_cols = [c for c in group_cols if c in joined.columns]

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
    return joined


def build_all_shares_targets(
    base_targets: pd.DataFrame,
    cached_allvars_path: Path,
    all_mappings: List[Tuple[str, str, int, int, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Build enhanced targets where ALL variables use TMD x SOI shares.

    Every targeted variable (except XTOT population) uses:
      area_target = TMD_national_sum × (area_SOI / national_SOI)

    This ensures area targets sum exactly to national TMD totals.

    Parameters
    ----------
    base_targets : pd.DataFrame
        Base targets from SOI data (used for geographic shares).
    cached_allvars_path : Path
        Path to ``cached_allvars.csv``.
    all_mappings : list of (tmdvar, soi_base, count, fstatus, desc)
        All variable/count/fstatus combinations to share.
    agi_cuts : list of float
        AGI bin cut points.

    Returns
    -------
    pd.DataFrame
        Enhanced targets with sort column, ready for target_file_writer.
    """
    # 1. Compute TMD national sums for all variables
    tmd_sums = compute_tmd_national_sums_all(
        cached_allvars_path, all_mappings, agi_cuts
    )

    # 2. Compute SOI geographic shares for all SOI base vars
    soi_shares = compute_soi_shares(base_targets, all_mappings)

    # Filter to only the exact (basesoivname, count, fstatus) combos
    # in all_mappings — SOI data may have extra count types
    wanted = pd.DataFrame(
        [
            {"basesoivname": m[1], "count": m[2], "fstatus": m[3]}
            for m in all_mappings
        ]
    ).drop_duplicates()
    soi_shares = soi_shares.merge(
        wanted,
        on=["basesoivname", "count", "fstatus"],
        how="inner",
    )

    # 3. Apply sharing formula
    shared = _apply_sharing(soi_shares, tmd_sums)

    # 4. Construct shared variable names
    shared["basesoivname"] = (
        "tmd"
        + shared["tmdvar"].str[1:]
        + "_shared_by_soi"
        + shared["basesoivname"]
    )
    shared["soivname"] = np.where(
        shared["count"] == 0,
        "a" + shared["basesoivname"],
        "n" + shared["basesoivname"],
    )

    # 5. Ensure area column exists
    if "area" not in shared.columns:
        shared["area"] = shared["stabbr"]

    # 6. Extract XTOT rows from base_targets
    xtot = base_targets.loc[base_targets["basesoivname"] == "XTOT"].copy()

    # 7. Select output columns
    out_cols = [
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
    ]
    shared_out = shared[[c for c in out_cols if c in shared.columns]]
    xtot_out = xtot[[c for c in out_cols if c in xtot.columns]]

    # 8. Stack XTOT + shared targets
    stack = pd.concat([xtot_out, shared_out], ignore_index=True)

    # 9. Sort: XTOT first, then everything else
    is_xtot = (stack["basesoivname"] == "XTOT") & (stack["scope"] == 0)
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
