"""
Congressional District SOI data ingestion and base targets construction.

Replaces the R/Quarto pipeline:
  - cd_construct_long_soi_data_file.qmd
  - cd_create_basefile_for_117Congress_cd_target_files.qmd

Pipeline:
  1. Read raw SOI CD CSV from ZIP archive.
  2. Classify record types (US, DC, cdstate, state, cd).
  3. Pivot to long format.
  4. Classify variables (vtype, basesoivname).
  5. Create derived variables (18400 = 18425 + 18450).
  6. Multiply amount values by 1000.
  7. Add AGI labels.
  8. For CD and cdstate records, add count/scope/fstatus metadata.
  9. Append XTOT (population) records.
  10. Produce ``base_targets`` DataFrame.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.areas.prepare.constants import (
    ALLCOUNT_VARS,
    AreaType,
    SOI_CD_CSV_IN_ZIP,
    SOI_CD_ZIP_PATTERNS,
    build_agi_labels,
)
from tmd.areas.prepare.soi_state_data import classify_soi_variable

# Single-CD states (always just one at-large district)
SINGLE_CD_STATES = {"AK", "DE", "DC", "MT", "ND", "SD", "VT", "WY"}


# ---- Read raw CD CSV from ZIP -----------------------------------


def read_soi_cd_csv(
    raw_data_dir: Path,
    year: int,
) -> pd.DataFrame:
    """
    Read raw SOI CD CSV from its ZIP archive.

    Parameters
    ----------
    raw_data_dir : Path
        Directory containing the ZIP file.
    year : int
        Data year (e.g. 2021, 2022).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with lowercase columns, stabbr, congdist,
        agistub, and year.
    """
    zip_name = SOI_CD_ZIP_PATTERNS.get(year)
    csv_name = SOI_CD_CSV_IN_ZIP.get(year)
    if zip_name is None or csv_name is None:
        raise FileNotFoundError(
            f"No SOI CD data for year {year}. "
            f"Available: {sorted(SOI_CD_ZIP_PATTERNS.keys())}"
        )
    zpath = raw_data_dir / zip_name
    if not zpath.exists():
        raise FileNotFoundError(f"CD ZIP not found: {zpath}")

    with zipfile.ZipFile(zpath) as zf:
        with zf.open(csv_name) as f:
            df = pd.read_csv(io.TextIOWrapper(f), thousands=",")

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(
        columns={
            "state": "stabbr",
            "agi_stub": "agistub",
            "cong_district": "congdist",
        }
    )
    # Ensure congdist is zero-padded 2-char string
    df["congdist"] = df["congdist"].astype(int).astype(str).str.zfill(2)
    df["year"] = year
    return df


# ---- Classify record types --------------------------------------


def classify_cd_records(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``rectype`` column classifying each row.

    Record types:
      - "US": U.S. total rows
      - "DC": District of Columbia rows
      - "cdstate": single-CD states (record is both state and CD)
      - "state": state aggregate row for multi-CD states (congdist=00)
      - "cd": individual CD row for multi-CD states
    """
    df = wide_df.copy()
    # Count agistub==0 records per state (determines single vs multi CD)
    nstub0 = (
        df.loc[df["agistub"] == 0]
        .groupby("stabbr")
        .size()
        .reset_index(name="nstub0")
    )
    df = df.merge(nstub0, on="stabbr", how="left")

    df["rectype"] = np.select(
        [
            df["stabbr"].isin(["US", "DC"]),
            df["nstub0"] == 1,
            (df["nstub0"] > 1) & (df["congdist"] == "00"),
            (df["nstub0"] > 1) & (df["congdist"] != "00"),
        ],
        ["US_or_DC", "cdstate", "state", "cd"],
        default="ERROR",
    )
    # Split US and DC
    df.loc[df["stabbr"] == "US", "rectype"] = "US"
    df.loc[df["stabbr"] == "DC", "rectype"] = "DC"

    df = df.drop(columns=["nstub0"])
    return df


# ---- Pivot to long format ---------------------------------------


def pivot_cd_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot wide CD SOI data to long format.

    Returns DataFrame with columns:
      rectype, stabbr, congdist, agistub, year, soivname, value
    """
    id_cols = [
        "rectype",
        "statefips",
        "stabbr",
        "congdist",
        "agistub",
        "year",
    ]
    value_cols = [c for c in wide_df.columns if c not in id_cols]
    long = wide_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="soivname",
        value_name="value",
    )
    long = long.dropna(subset=["value"]).copy()
    return long


# ---- Derived variables ------------------------------------------


def create_cd_derived_variables(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived SOI variables for CD data.

    Creates a18400/n18400 = a18425 + a18450 / n18425 + n18450.
    """
    mask = long_df["soivname"].str[1:].isin(["18425", "18450"])
    components = long_df.loc[mask].copy()
    if components.empty:
        return long_df
    components["soivname"] = components["soivname"].str[0] + "18400"
    group_cols = [
        "rectype",
        "stabbr",
        "congdist",
        "agistub",
        "year",
        "soivname",
    ]
    derived = components.groupby(group_cols, as_index=False)["value"].sum()
    return pd.concat([long_df, derived], ignore_index=True)


# ---- Annotate and scale -----------------------------------------


def annotate_cd_variables(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add basesoivname and vtype by classifying soivname."""
    classifications = long_df["soivname"].apply(classify_soi_variable)
    df = long_df.copy()
    df["basesoivname"] = classifications.str[0]
    df["vtype"] = classifications.str[1]
    return df


def scale_cd_amounts(long_df: pd.DataFrame) -> pd.DataFrame:
    """Multiply amount values by 1000 (SOI stores in thousands)."""
    df = long_df.copy()
    mask = df["vtype"] == "amount"
    df.loc[mask, "value"] = df.loc[mask, "value"] * 1000
    return df


# ---- Full pipeline: ZIP to soilong DataFrame --------------------


def create_cd_soilong(
    raw_data_dir: Path,
    year: int,
) -> pd.DataFrame:
    """
    Full pipeline from raw SOI CD ZIP to annotated long DataFrame.

    Steps: read ZIP → classify records → pivot → derive → classify
    → scale amounts → add labels.
    """
    wide = read_soi_cd_csv(raw_data_dir, year)
    wide = classify_cd_records(wide)
    long = pivot_cd_to_long(wide)
    long = create_cd_derived_variables(long)
    long = annotate_cd_variables(long)
    long = scale_cd_amounts(long)
    # Add AGI labels
    agi_labels = build_agi_labels(AreaType.CD)
    long = long.merge(
        agi_labels[["agistub", "agilo", "agihi", "agilabel"]],
        on="agistub",
        how="left",
    )
    long = long.sort_values(
        ["rectype", "stabbr", "congdist", "soivname", "agistub"]
    ).reset_index(drop=True)
    return long


# ---- Base targets construction ----------------------------------


def create_cd_base_targets(
    soilong: pd.DataFrame,
    pop_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create CD base_targets from SOI long data and population.

    Reproduces cd_create_basefile_for_117Congress_cd_target_files.qmd:
      1. Filter to CD/cdstate/DC/US records.
      2. Add scope, count, fstatus metadata.
      3. Reconstruct US totals by summing across CDs.
      4. Create XTOT population records.
      5. Build area code = stabbr + congdist.
      6. Stack and sort.

    Parameters
    ----------
    soilong : pd.DataFrame
        Output of ``create_cd_soilong()``.
    pop_df : pd.DataFrame
        CD population data with columns (stabbr, congdist, population).

    Returns
    -------
    pd.DataFrame
        Base targets with columns: stabbr, area, count, scope,
        agilo, agihi, fstatus, target, basesoivname, soivname,
        agistub, agilabel.
    """
    # 1. Keep only CD-relevant records (drop state aggregates)
    soi = soilong.loc[
        soilong["rectype"].isin(["US", "cd", "cdstate", "DC"])
    ].copy()

    # 2. Add scope (always 1 = PUF-derived)
    soi["scope"] = 1

    # 3. Assign count type
    soi["count"] = np.where(
        soi["vtype"] == "amount",
        0,
        np.where(
            soi["soivname"].isin(ALLCOUNT_VARS),
            1,
            2,
        ),
    )

    # 4. Assign filing status
    mars_mask = soi["soivname"].str.startswith("mars")
    soi["fstatus"] = 0
    soi.loc[mars_mask, "fstatus"] = (
        soi.loc[mars_mask, "soivname"].str[-1].astype(int)
    )

    # 5. Rename value → target
    soi = soi.rename(columns={"value": "target"})

    # 6. Build area code
    soi["area"] = soi["stabbr"] + soi["congdist"]

    # 7. Reconstruct US totals by summing across CDs
    # (The US record in SOI includes all areas; we reconstruct from
    # CD-level to match the R pipeline which also does this)
    non_us = soi.loc[soi["stabbr"] != "US"].copy()
    us_sums = (
        non_us.loc[non_us["rectype"].isin(["cd", "cdstate", "DC"])]
        .groupby(
            [
                "agistub",
                "agilo",
                "agihi",
                "agilabel",
                "soivname",
                "basesoivname",
                "scope",
                "fstatus",
                "count",
            ],
            as_index=False,
        )["target"]
        .sum()
    )
    us_sums["stabbr"] = "US"
    us_sums["congdist"] = "00"
    us_sums["area"] = "US00"
    us_sums["rectype"] = "US"

    # Replace original US rows with reconstructed sums
    soi = pd.concat(
        [non_us, us_sums],
        ignore_index=True,
    )

    soi = soi[
        [
            "stabbr",
            "congdist",
            "area",
            "soivname",
            "basesoivname",
            "count",
            "scope",
            "agilo",
            "agihi",
            "fstatus",
            "target",
            "agistub",
            "agilabel",
        ]
    ].copy()

    # 8. Create XTOT (population) records
    agi_labels = build_agi_labels(AreaType.CD)
    agi0 = agi_labels.loc[agi_labels["agistub"] == 0].iloc[0]
    pop_recs = pop_df.copy()
    pop_recs["soivname"] = "XTOT"
    pop_recs["basesoivname"] = "XTOT"
    pop_recs["agistub"] = 0
    pop_recs["count"] = 0
    pop_recs["scope"] = 0
    pop_recs["fstatus"] = 0
    pop_recs["target"] = pop_recs["population"]
    pop_recs["agilo"] = agi0["agilo"]
    pop_recs["agihi"] = agi0["agihi"]
    pop_recs["agilabel"] = agi0["agilabel"]
    pop_recs["area"] = pop_recs["stabbr"] + pop_recs["congdist"]
    pop_recs = pop_recs[soi.columns].copy()

    # 9. Combine and sort
    base_targets = pd.concat([pop_recs, soi], ignore_index=True)
    base_targets = base_targets.sort_values(
        [
            "stabbr",
            "scope",
            "fstatus",
            "basesoivname",
            "count",
            "agistub",
        ]
    ).reset_index(drop=True)

    return base_targets
