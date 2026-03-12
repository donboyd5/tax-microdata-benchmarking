"""
Census state and CD population data for area target preparation.

Provides population estimates from the Census Bureau's Population
Estimates Program (PEP).  Default data is embedded (vintage 2021);
a user-supplied CSV overrides the embedded values.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# --- 2021 Population Estimates (PEP vintage 2021) ---
# Source: U.S. Census Bureau, Population Estimates Program
# https://api.census.gov/data/2021/pep/population
# 52 states/territories + US total

_STATE_POP_2021 = {
    "AK": 732673,
    "AL": 5039877,
    "AR": 3025891,
    "AZ": 7276316,
    "CA": 39237836,
    "CO": 5812069,
    "CT": 3605597,
    "DC": 670050,
    "DE": 1003384,
    "FL": 21781128,
    "GA": 10799566,
    "HI": 1441553,
    "IA": 3193079,
    "ID": 1900923,
    "IL": 12671469,
    "IN": 6805985,
    "KS": 2934582,
    "KY": 4509394,
    "LA": 4624047,
    "MA": 6984723,
    "MD": 6165129,
    "ME": 1372247,
    "MI": 10050811,
    "MN": 5707390,
    "MO": 6168187,
    "MS": 2949965,
    "MT": 1104271,
    "NC": 10551162,
    "ND": 774948,
    "NE": 1963692,
    "NH": 1388992,
    "NJ": 9267130,
    "NM": 2115877,
    "NV": 3143991,
    "NY": 19835913,
    "OH": 11780017,
    "OK": 3986639,
    "OR": 4246155,
    "PA": 12964056,
    "PR": 3263584,
    "RI": 1095610,
    "SC": 5190705,
    "SD": 895376,
    "TN": 6975218,
    "TX": 29527941,
    "US": 335157329,
    "UT": 3337975,
    "VA": 8642274,
    "VT": 645570,
    "WA": 7738692,
    "WI": 5895908,
    "WV": 1782959,
    "WY": 578803,
}

_EMBEDDED_DATA = {
    2021: _STATE_POP_2021,
}


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
        overrides the embedded data.

    Returns
    -------
    pd.DataFrame
        Columns: stabbr (str), population (int).
        Includes 50 states, DC, PR, and US.
    """
    if csv_path is not None:
        return _read_population_csv(csv_path, year)
    if year in _EMBEDDED_DATA:
        pop = _EMBEDDED_DATA[year]
        df = pd.DataFrame(list(pop.items()), columns=["stabbr", "population"])
        return df.sort_values("stabbr").reset_index(drop=True)
    raise ValueError(
        f"No embedded population data for {year}. "
        f"Available years: {sorted(_EMBEDDED_DATA.keys())}. "
        f"Supply a csv_path to use custom data."
    )


def _read_population_csv(csv_path: Path, year: int) -> pd.DataFrame:
    """Read population CSV, normalise column names."""
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
