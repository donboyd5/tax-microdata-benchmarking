"""
Census state and CD population data for area target preparation.

Provides population estimates:
  - States: Census Bureau Population Estimates Program (PEP).
  - CDs: American Community Survey 1-year estimates.

Default data is stored in JSON files under ``data/``; a user-supplied
CSV overrides the defaults.

Note on CD boundaries:
  - 2021 CD populations use 117th Congress boundaries (ACS 2021 1-year).
  - 2022 CD populations use 118th Congress boundaries (ACS 2022 1-year).
  - Both 2021 and 2022 SOI CD data use 117th Congress boundaries,
    so 2021 CD populations are used for 117th Congress processing
    regardless of SOI year. The 2022 CD populations (118th Congress)
    are used after the crosswalk converts targets to 118th boundaries.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def _load_population_json(filename: str) -> Dict[str, Dict[str, int]]:
    """Load a population JSON file and return {year_str: {area: pop}}."""
    path = _DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filter out metadata keys (those starting with _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_state_population(
    year: int,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return state population as DataFrame with columns (stabbr, population).

    Parameters
    ----------
    year : int
        Calendar year for the population estimate.
    csv_path : Path, optional
        Path to a CSV with columns ``stabbr`` and a population column
        (named ``pop{year}`` or ``population``).  If provided, this
        overrides the default data.

    Returns
    -------
    pd.DataFrame
        Columns: stabbr (str), population (int).
        Includes 50 states, DC, PR, and US.
    """
    if csv_path is not None:
        return _read_population_csv(csv_path, year)
    all_years = _load_population_json("state_populations.json")
    year_str = str(year)
    if year_str not in all_years:
        raise ValueError(
            f"No state population data for {year}. "
            f"Available years: {sorted(all_years.keys())}. "
            f"Supply a csv_path to use custom data."
        )
    pop = all_years[year_str]
    df = pd.DataFrame(list(pop.items()), columns=["stabbr", "population"])
    df["population"] = df["population"].astype(int)
    return df.sort_values("stabbr").reset_index(drop=True)


def get_cd_population(
    year: int,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return CD population as DataFrame.

    Parameters
    ----------
    year : int
        Calendar year for the population estimate.
    csv_path : Path, optional
        Path to a CSV with columns ``stabbr``, ``congdist``, and
        a population column.  If provided, overrides default data.

    Returns
    -------
    pd.DataFrame
        Columns: stabbr (str), congdist (str), population (int).
        Includes all CDs + a US00 total row.
    """
    if csv_path is not None:
        return _read_cd_population_csv(csv_path, year)
    all_years = _load_population_json("cd_populations.json")
    year_str = str(year)
    if year_str not in all_years:
        raise ValueError(
            f"No CD population data for {year}. "
            f"Available years: {sorted(all_years.keys())}. "
            f"Supply a csv_path to use custom data."
        )
    pop = all_years[year_str]
    rows = []
    for area_code, population in pop.items():
        stabbr = area_code[:2]
        congdist = area_code[2:]
        rows.append(
            {
                "stabbr": stabbr,
                "congdist": congdist,
                "population": int(population),
            }
        )
    df = pd.DataFrame(rows)
    # Add US total row
    us_total = df["population"].sum()
    us_row = pd.DataFrame(
        [{"stabbr": "US", "congdist": "00", "population": us_total}]
    )
    df = pd.concat([df, us_row], ignore_index=True)
    return df.sort_values(["stabbr", "congdist"]).reset_index(drop=True)


def _read_population_csv(csv_path: Path, year: int) -> pd.DataFrame:
    """Read state population CSV, normalise column names."""
    df = pd.read_csv(csv_path)
    # Accept either pop{year} or population as column name
    pop_col = f"pop{year}"
    if pop_col in df.columns:
        df = df.rename(columns={pop_col: "population"})
    elif "population" not in df.columns:
        # Try any column that looks like pop*
        pop_cols = [c for c in df.columns if c.startswith("pop")]
        if len(pop_cols) == 1:
            df = df.rename(columns={pop_cols[0]: "population"})
        else:
            raise ValueError(
                f"Cannot find population column in {csv_path}. "
                f"Expected 'pop{year}' or 'population'."
            )
    df = df[["stabbr", "population"]].copy()
    df["population"] = df["population"].astype(int)
    return df.sort_values("stabbr").reset_index(drop=True)


def _read_cd_population_csv(csv_path: Path, year: int) -> pd.DataFrame:
    """Read CD population CSV, normalise column names."""
    df = pd.read_csv(csv_path)
    pop_col = f"pop{year}"
    if pop_col in df.columns:
        df = df.rename(columns={pop_col: "population"})
    elif "population" not in df.columns:
        pop_cols = [c for c in df.columns if c.startswith("pop")]
        if len(pop_cols) == 1:
            df = df.rename(columns={pop_cols[0]: "population"})
        else:
            raise ValueError(
                f"Cannot find population column in {csv_path}. "
                f"Expected 'pop{year}' or 'population'."
            )
    req_cols = ["stabbr", "congdist", "population"]
    # Some CSVs may use different column names
    if "congdist" not in df.columns:
        for alt in ["CONG_DISTRICT", "cong_district", "cd"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "congdist"})
                break
    df = df[req_cols].copy()
    df["population"] = df["population"].astype(int)
    return df.sort_values(["stabbr", "congdist"]).reset_index(drop=True)
