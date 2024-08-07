"""
This module adds basic sanity tests for each TC-input flat file,
checking that the totals of each variable are within the ballpark
of the Tax-Data 2023 PUF's totals.
"""

import os
import pytest
import yaml
from pathlib import Path
import pytest
import pandas as pd
import subprocess
import warnings
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.create_taxcalc_input_variables import (
    create_variable_file,
)
from tax_microdata_benchmarking.utils.taxcalc_utils import (
    get_tax_expenditure_results,
)

warnings.filterwarnings("ignore")

test_mode = os.environ.get("TEST_MODE", "lite")

FOLDER = Path(__file__).parent

with open(FOLDER / "tc_variable_totals.yaml") as f:
    tc_variable_totals = yaml.safe_load(f)

with open(FOLDER / "tax_expenditure_targets.yaml") as f:
    tax_expenditure_targets = yaml.safe_load(f)

with open(STORAGE_FOLDER / "input" / "taxcalc_variable_metadata.yaml") as f:
    taxcalc_variable_metadata = yaml.safe_load(f)

tmd_weights_path = STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"

tmd_growfactors_path = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"

EXEMPTED_VARIABLES = [
    "DSI",  # Issue here but deprioritized.
    "EIC",  # PUF-PE file almost certainly more correct by including CPS data
    "MIDR",  # Issue here but deprioritized.
    "RECID",  # No reason to compare.
    "a_lineno",  # No reason to compare.
    "agi_bin",  # No reason to compare.
    "blind_spouse",  # Issue here but deprioritized.
    "cmbtp",  # No reason to compare.
    "data_source",  # No reason to compare.
    "s006",  # No reason to compare.
    "h_seq",  # No reason to compare.
    "fips",  # No reason to compare.
    "ffpos",  # No reason to compare.
    "p22250",  # PE-PUF likely closer to truth than taxdata (needs triple check).
    "p23250",  # PE-PUF likely closer to truth than taxdata (needs triple check).
    "e01200",  # Unknown but deprioritized for now.
    "e17500",  # Unknown but deprioritized for now.
    "e18500",  # Unknown but deprioritized for now.
    "e02100",  # Farm income, unsure who's closer.
    "e02300",  # UI exploded in 2021
    "e02400",  # SS benefits, TD is out
    "e18400",
    "e19200",
    "e20100",
]

# Exempt any variable split between filer and spouse for now.
EXEMPTED_VARIABLES += [
    variable
    for variable in taxcalc_variable_metadata["read"]
    if variable.endswith("p") or variable.endswith("s")
]


variables_to_test = [
    variable
    for variable in tc_variable_totals.keys()
    if variable not in EXEMPTED_VARIABLES
]

dataset_names_to_test = ["tmd_2021"]

datasets_to_test = [
    pd.read_csv(STORAGE_FOLDER / "output" / f"{dataset}.csv")
    for dataset in dataset_names_to_test
]


@pytest.mark.vartotals
@pytest.mark.parametrize("variable", variables_to_test, ids=lambda x: x)
@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_variable_totals(variable, flat_file):
    meta = taxcalc_variable_metadata["read"][variable]
    name = meta.get("desc")
    weight = flat_file.s006_original
    total = (flat_file[variable] * weight * (flat_file.data_source == 1)).sum()
    if tc_variable_totals[variable] == 0:
        # If the taxdata file has a zero total, we'll assume the tested file is correct in the absence of better data.
        return
    # 45% and more than $30bn off taxdata is a failure.
    assert (
        abs(total / tc_variable_totals[variable] - 1) < 0.45
        or abs(total / 1e9 - tc_variable_totals[variable] / 1e9) < 30
    ), f"{variable} ({name}) differs to tax-data by {total / tc_variable_totals[variable] - 1:.1%} ({total/1e9:.1f}bn vs {tc_variable_totals[variable]/1e9:.1f}bn)"


tax_expenditure_reforms = [
    "ctc",
    "eitc",
    "social_security_partial_taxability",
    "niit",
    "cgqd_tax_preference",
    "qbid",
    "salt",
]


tax_expenditure_estimates = {}

for dataset, name in zip(datasets_to_test, dataset_names_to_test):
    print(f"Running tax expenditure estimates for {dataset}")
    tax_expenditure_estimates[name] = get_tax_expenditure_results(
        dataset,
        2021,  # dataset year
        2023,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )
    assert len(datasets_to_test) == 1
    _ = get_tax_expenditure_results(
        datasets_to_test[0],
        2021,  # dataset year
        2026,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )


@pytest.mark.taxexp
@pytest.mark.parametrize("flat_file", dataset_names_to_test, ids=lambda x: x)
@pytest.mark.parametrize("reform", tax_expenditure_reforms, ids=lambda x: x)
def test_tax_expenditure_estimates(
    flat_file: pd.DataFrame,
    reform: str,
):
    target = tax_expenditure_targets[reform][2023]
    estimate = tax_expenditure_estimates[flat_file][reform]
    tol = 0.4
    if reform == "salt":
        tol = 0.7
    assert abs(estimate / target - 1) < tol or abs(estimate - target) < tol, (
        f"{reform} differs from CBO estimate by "
        f"{estimate / target - 1:.1%} ({estimate:.1f}bn vs {target:.1f}bn)"
    )


def test_create_taxcalc_tmd_file():
    create_variable_file(write_file=False)


@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_no_negative_weights(flat_file):
    assert flat_file.s006.min() >= 0, "Negative weights found."


@pytest.mark.qbid
@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_qbided_close_to_soi(flat_file):
    assert (
        abs((flat_file.s006 * flat_file.qbided).sum() / 1e9 / 205.8 - 1) < 0.25
    ), "QBIDED not within 25 percent of 205.8bn"


@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_partnership_s_corp_income_close_to_soi(flat_file):
    assert (
        abs((flat_file.s006 * flat_file.e26270).sum() / 1e9 / 975 - 1) < 0.1
    ), "Partnership/S-Corp income not within 10 percent of 975bn"
