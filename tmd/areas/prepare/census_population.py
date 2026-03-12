"""
Census state and CD population data for area target preparation.

Provides population estimates:
  - States: Census Bureau Population Estimates Program (PEP).
  - CDs: American Community Survey 1-year estimates.

Default data is embedded (vintage 2021); a user-supplied CSV
overrides the embedded values.
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

_EMBEDDED_STATE_DATA = {
    2021: _STATE_POP_2021,
}

# --- 2021 CD Population (ACS 1-year estimates) ---
# Source: U.S. Census Bureau, ACS 1-year estimates, table B01003
# https://api.census.gov/data/2021/acs/acs1
# 437 CDs (117th Congress boundaries) + US total
# DC uses code DC00 (Census returns DC98).

_CD_POP_2021 = {
    "AK00": 732673,
    "AL01": 735373,
    "AL02": 690107,
    "AL03": 742147,
    "AL04": 708532,
    "AL05": 773212,
    "AL06": 723686,
    "AL07": 666820,
    "AR01": 714143,
    "AR02": 771880,
    "AR03": 851053,
    "AR04": 688815,
    "AZ01": 774354,
    "AZ02": 754647,
    "AZ03": 780337,
    "AZ04": 848908,
    "AZ05": 891118,
    "AZ06": 786168,
    "AZ07": 791774,
    "AZ08": 863775,
    "AZ09": 785235,
    "CA01": 701063,
    "CA02": 718090,
    "CA03": 762445,
    "CA04": 782605,
    "CA05": 710312,
    "CA06": 784724,
    "CA07": 789764,
    "CA08": 759705,
    "CA09": 808663,
    "CA10": 774782,
    "CA11": 771855,
    "CA12": 718792,
    "CA13": 761542,
    "CA14": 719826,
    "CA15": 784297,
    "CA16": 751686,
    "CA17": 771306,
    "CA18": 721174,
    "CA19": 722407,
    "CA20": 748911,
    "CA21": 718991,
    "CA22": 810279,
    "CA23": 770688,
    "CA24": 740591,
    "CA25": 738117,
    "CA26": 717217,
    "CA27": 688082,
    "CA28": 694675,
    "CA29": 691770,
    "CA30": 764996,
    "CA31": 758538,
    "CA32": 689035,
    "CA33": 686798,
    "CA34": 696606,
    "CA35": 748055,
    "CA36": 761959,
    "CA37": 700635,
    "CA38": 691366,
    "CA39": 729834,
    "CA40": 681700,
    "CA41": 765124,
    "CA42": 841945,
    "CA43": 728012,
    "CA44": 698216,
    "CA45": 808871,
    "CA46": 701726,
    "CA47": 701834,
    "CA48": 712717,
    "CA49": 740582,
    "CA50": 748539,
    "CA51": 707085,
    "CA52": 778194,
    "CA53": 761110,
    "CO01": 838929,
    "CO02": 820942,
    "CO03": 769771,
    "CO04": 908778,
    "CO05": 840942,
    "CO06": 833948,
    "CO07": 798759,
    "CT01": 716422,
    "CT02": 706620,
    "CT03": 711522,
    "CT04": 745760,
    "CT05": 725273,
    "DC00": 670050,
    "DE00": 1003384,
    "FL01": 821362,
    "FL02": 749149,
    "FL03": 774718,
    "FL04": 893302,
    "FL05": 747582,
    "FL06": 814531,
    "FL07": 787989,
    "FL08": 793922,
    "FL09": 1002804,
    "FL10": 859992,
    "FL11": 850097,
    "FL12": 829889,
    "FL13": 723346,
    "FL14": 810520,
    "FL15": 820172,
    "FL16": 907070,
    "FL17": 802733,
    "FL18": 809802,
    "FL19": 864808,
    "FL20": 761114,
    "FL21": 797947,
    "FL22": 783681,
    "FL23": 765245,
    "FL24": 735549,
    "FL25": 761486,
    "FL26": 784436,
    "FL27": 727882,
    "GA01": 765267,
    "GA02": 672932,
    "GA03": 773330,
    "GA04": 775516,
    "GA05": 789610,
    "GA06": 760368,
    "GA07": 856376,
    "GA08": 717457,
    "GA09": 805448,
    "GA10": 800510,
    "GA11": 807381,
    "GA12": 742856,
    "GA13": 795514,
    "GA14": 737001,
    "HI01": 715138,
    "HI02": 726415,
    "IA01": 776069,
    "IA02": 785047,
    "IA03": 867614,
    "IA04": 764349,
    "ID01": 1007769,
    "ID02": 893154,
    "IL01": 676454,
    "IL02": 681277,
    "IL03": 722714,
    "IL04": 640957,
    "IL05": 750823,
    "IL06": 754166,
    "IL07": 747262,
    "IL08": 701275,
    "IL09": 735119,
    "IL10": 719131,
    "IL11": 717925,
    "IL12": 667669,
    "IL13": 690804,
    "IL14": 736577,
    "IL15": 675647,
    "IL16": 686140,
    "IL17": 665076,
    "IL18": 702453,
    "IN01": 735140,
    "IN02": 729230,
    "IN03": 761500,
    "IN04": 769259,
    "IN05": 822535,
    "IN06": 721123,
    "IN07": 775837,
    "IN08": 717065,
    "IN09": 774296,
    "KS01": 697903,
    "KS02": 711743,
    "KS03": 794125,
    "KS04": 730811,
    "KY01": 713098,
    "KY02": 786202,
    "KY03": 751181,
    "KY04": 783484,
    "KY05": 689987,
    "KY06": 785442,
    "LA01": 816893,
    "LA02": 756483,
    "LA03": 771394,
    "LA04": 722852,
    "LA05": 728573,
    "LA06": 827852,
    "MA01": 730798,
    "MA02": 779662,
    "MA03": 790963,
    "MA04": 767979,
    "MA05": 781372,
    "MA06": 774905,
    "MA07": 779197,
    "MA08": 792652,
    "MA09": 787195,
    "MD01": 739031,
    "MD02": 766679,
    "MD03": 752494,
    "MD04": 788525,
    "MD05": 804124,
    "MD06": 791771,
    "MD07": 726949,
    "MD08": 795556,
    "ME01": 716258,
    "ME02": 655989,
    "MI01": 707596,
    "MI02": 761672,
    "MI03": 756944,
    "MI04": 698588,
    "MI05": 664225,
    "MI06": 720214,
    "MI07": 715356,
    "MI08": 742257,
    "MI09": 711987,
    "MI10": 725418,
    "MI11": 757991,
    "MI12": 741026,
    "MI13": 674738,
    "MI14": 672799,
    "MN01": 691635,
    "MN02": 738835,
    "MN03": 734894,
    "MN04": 721132,
    "MN05": 727716,
    "MN06": 744738,
    "MN07": 666533,
    "MN08": 681907,
    "MO01": 710566,
    "MO02": 763765,
    "MO03": 817915,
    "MO04": 783543,
    "MO05": 783027,
    "MO06": 788813,
    "MO07": 799499,
    "MO08": 721059,
    "MS01": 764049,
    "MS02": 660002,
    "MS03": 745609,
    "MS04": 780305,
    "MT00": 1104271,
    "NC01": 732511,
    "NC02": 938080,
    "NC03": 760997,
    "NC04": 885785,
    "NC05": 760703,
    "NC06": 814959,
    "NC07": 795070,
    "NC08": 831474,
    "NC09": 770733,
    "NC10": 783974,
    "NC11": 775259,
    "NC12": 909148,
    "NC13": 792469,
    "ND00": 774948,
    "NE01": 662804,
    "NE02": 706307,
    "NE03": 594581,
    "NH01": 708467,
    "NH02": 680525,
    "NJ01": 749827,
    "NJ02": 739987,
    "NJ03": 761771,
    "NJ04": 804956,
    "NJ05": 757022,
    "NJ06": 762208,
    "NJ07": 761486,
    "NJ08": 797874,
    "NJ09": 778948,
    "NJ10": 808433,
    "NJ11": 761290,
    "NJ12": 783328,
    "NM01": 698553,
    "NM02": 712769,
    "NM03": 704555,
    "NV01": 686299,
    "NV02": 766406,
    "NV03": 883132,
    "NV04": 808154,
    "NY01": 738887,
    "NY02": 730149,
    "NY03": 746449,
    "NY04": 741844,
    "NY05": 819686,
    "NY06": 733056,
    "NY07": 680802,
    "NY08": 779894,
    "NY09": 776441,
    "NY10": 757587,
    "NY11": 749691,
    "NY12": 711858,
    "NY13": 748859,
    "NY14": 689682,
    "NY15": 732872,
    "NY16": 753052,
    "NY17": 763172,
    "NY18": 748718,
    "NY19": 704391,
    "NY20": 742207,
    "NY21": 707234,
    "NY22": 689366,
    "NY23": 687238,
    "NY24": 715194,
    "NY25": 728525,
    "NY26": 734798,
    "NY27": 724261,
    "OH01": 766737,
    "OH02": 738776,
    "OH03": 809881,
    "OH04": 714581,
    "OH05": 731655,
    "OH06": 681651,
    "OH07": 743551,
    "OH08": 745908,
    "OH09": 704330,
    "OH10": 728978,
    "OH11": 676238,
    "OH12": 808118,
    "OH13": 704120,
    "OH14": 719537,
    "OH15": 769052,
    "OH16": 736904,
    "OK01": 836922,
    "OK02": 724389,
    "OK03": 790472,
    "OK04": 805943,
    "OK05": 828913,
    "OR01": 864740,
    "OR02": 864256,
    "OR03": 841904,
    "OR04": 825543,
    "OR05": 849712,
    "PA01": 730085,
    "PA02": 737973,
    "PA03": 729976,
    "PA04": 758187,
    "PA05": 716820,
    "PA06": 757071,
    "PA07": 756023,
    "PA08": 709790,
    "PA09": 709976,
    "PA10": 758088,
    "PA11": 755278,
    "PA12": 684344,
    "PA13": 694541,
    "PA14": 684640,
    "PA15": 667819,
    "PA16": 687347,
    "PA17": 732334,
    "PA18": 693764,
    "RI01": 551922,
    "RI02": 543688,
    "SC01": 824201,
    "SC02": 726470,
    "SC03": 708699,
    "SC04": 780255,
    "SC05": 747217,
    "SC06": 660275,
    "SC07": 743588,
    "SD00": 895376,
    "TN01": 734581,
    "TN02": 784818,
    "TN03": 745925,
    "TN04": 850783,
    "TN05": 790094,
    "TN06": 828312,
    "TN07": 838746,
    "TN08": 714283,
    "TN09": 687676,
    "TX01": 728600,
    "TX02": 833269,
    "TX03": 968602,
    "TX04": 810364,
    "TX05": 797242,
    "TX06": 833987,
    "TX07": 817706,
    "TX08": 944280,
    "TX09": 765427,
    "TX10": 993729,
    "TX11": 779873,
    "TX12": 889755,
    "TX13": 710745,
    "TX14": 774332,
    "TX15": 816197,
    "TX16": 765065,
    "TX17": 815271,
    "TX18": 781929,
    "TX19": 738129,
    "TX20": 790332,
    "TX21": 877749,
    "TX22": 1034272,
    "TX23": 847567,
    "TX24": 815338,
    "TX25": 876198,
    "TX26": 972162,
    "TX27": 749930,
    "TX28": 774870,
    "TX29": 685520,
    "TX30": 789579,
    "TX31": 969723,
    "TX32": 774999,
    "TX33": 721392,
    "TX34": 715023,
    "TX35": 799047,
    "TX36": 769738,
    "UT01": 821989,
    "UT02": 811531,
    "UT03": 798985,
    "UT04": 905470,
    "VA01": 836319,
    "VA02": 748533,
    "VA03": 761402,
    "VA04": 781353,
    "VA05": 742839,
    "VA06": 763436,
    "VA07": 830040,
    "VA08": 788119,
    "VA09": 693122,
    "VA10": 895085,
    "VA11": 802026,
    "VT00": 645570,
    "WA01": 801211,
    "WA02": 767914,
    "WA03": 781241,
    "WA04": 752581,
    "WA05": 763080,
    "WA06": 738969,
    "WA07": 799770,
    "WA08": 783018,
    "WA09": 760454,
    "WA10": 790454,
    "WI01": 722922,
    "WI02": 793162,
    "WI03": 733081,
    "WI04": 685822,
    "WI05": 741458,
    "WI06": 727054,
    "WI07": 739235,
    "WI08": 753174,
    "WV01": 600701,
    "WV02": 620298,
    "WV03": 561960,
    "WY00": 578803,
}

_EMBEDDED_CD_DATA = {
    2021: _CD_POP_2021,
}

_EMBEDDED_STATE_DATA = {
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
    if year in _EMBEDDED_STATE_DATA:
        pop = _EMBEDDED_STATE_DATA[year]
        df = pd.DataFrame(list(pop.items()), columns=["stabbr", "population"])
        return df.sort_values("stabbr").reset_index(drop=True)
    raise ValueError(
        f"No embedded state population data for {year}. "
        f"Available years: {sorted(_EMBEDDED_STATE_DATA.keys())}. "
        f"Supply a csv_path to use custom data."
    )


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
        a population column.  If provided, overrides embedded data.

    Returns
    -------
    pd.DataFrame
        Columns: stabbr (str), congdist (str), population (int).
        Includes all CDs + a US00 total row.
    """
    if csv_path is not None:
        return _read_cd_population_csv(csv_path, year)
    if year in _EMBEDDED_CD_DATA:
        pop = _EMBEDDED_CD_DATA[year]
        rows = []
        for area_code, population in pop.items():
            stabbr = area_code[:2]
            congdist = area_code[2:]
            rows.append(
                {
                    "stabbr": stabbr,
                    "congdist": congdist,
                    "population": population,
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
    raise ValueError(
        f"No embedded CD population data for {year}. "
        f"Available years: {sorted(_EMBEDDED_CD_DATA.keys())}. "
        f"Supply a csv_path to use custom data."
    )


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
