"""
Tests of income/payroll tax revenues generated by the tmd files.
"""

import yaml
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER


FIRST_CYR = 2021
LAST_CYR = 2033

DEFAULT_RELTOL_ITAX = 0.010
RELTOL_ITAX = {
    2021: 0.095,
    2022: 0.090,
    2024: 0.060,
    2025: 0.050,
    2026: 0.100,
    2027: 0.060,
    2028: 0.070,
    2029: 0.075,
    2030: 0.080,
    2031: 0.085,
    2032: 0.090,
    2033: 0.090,
}
DEFAULT_RELTOL_PTAX = 0.010
RELTOL_PTAX = {
    2021: 0.095,
    2022: 0.020,
    2024: 0.020,
}


DUMP = False  # True implies test always fails with complete output


def fy2cy(fy1, fy2):
    return fy1 + 0.25 * (fy2 - fy1)


def test_tax_revenue(
    tests_folder, tmd_variables, tmd_weights_path, tmd_growfactors_path
):
    # read expected fiscal year revenues and convert to calendar year revenues
    epath = tests_folder / "expected_itax_revenue.yaml"
    with open(epath, "r", encoding="utf-8") as f:
        fy_itax = yaml.safe_load(f)
    epath = tests_folder / "expected_ptax_revenue.yaml"
    with open(epath, "r", encoding="utf-8") as f:
        fy_ptax = yaml.safe_load(f)
    exp_itax = {}
    exp_ptax = {}
    for year in range(FIRST_CYR, LAST_CYR + 1):
        exp_itax[year] = round(fy2cy(fy_itax[year], fy_itax[year + 1]), 3)
        exp_ptax[year] = round(fy2cy(fy_ptax[year], fy_ptax[year + 1]), 3)
    # calculate actual tax revenues for each calendar year
    pol = tc.Policy.tmd_constructor(
        growfactors_path=(STORAGE_FOLDER / "output" / "tmd_growfactors.csv"),
    )
    wghts = str(tmd_weights_path)
    growf = tc.GrowFactors(growfactors_filename=str(tmd_growfactors_path))
    input_data = tc.Records(
        data=tmd_variables,
        start_year=2021,
        weights=wghts,
        gfactors=growf,
        adjust_ratios=None,
        exact_calculations=True,
        weights_scale=1.0,
    )
    sim = tc.Calculator(policy=pol, records=input_data)
    act_itax = {}
    act_ptax = {}
    for year in range(FIRST_CYR, LAST_CYR + 1):
        sim.advance_to_year(year)
        sim.calc_all()
        wght = sim.array("s006")
        itax = sim.array("iitax")  # includes refundable credit amounts
        refc = sim.array("refund")  # refundable credits considered expenditure
        itax_cbo = itax + refc  # itax revenue comparable to CBO estimates
        act_itax[year] = (wght * itax_cbo).sum() * 1e-9
        act_ptax[year] = (wght * sim.array("payrolltax")).sum() * 1e-9
    # compare actual vs expected tax revenues in each calendar year
    emsg = ""
    for year in range(FIRST_CYR, LAST_CYR + 1):
        reldiff = act_itax[year] / exp_itax[year] - 1
        same = abs(reldiff) < RELTOL_ITAX.get(year, DEFAULT_RELTOL_ITAX)
        if not same or DUMP:
            msg = (
                f"\nITAX:cyr,act,exp,rdiff= {year} "
                f"{act_itax[year]:9.3f} {exp_itax[year]:9.3f} {reldiff:7.4f}"
            )
            emsg += msg
        reldiff = act_ptax[year] / exp_ptax[year] - 1
        same = abs(reldiff) < RELTOL_PTAX.get(year, DEFAULT_RELTOL_PTAX)
        if not same or DUMP:
            msg = (
                f"\nPTAX:cyr,act,exp,rdiff= {year} "
                f"{act_ptax[year]:9.3f} {exp_ptax[year]:9.3f} {reldiff:7.4f}"
            )
            emsg += msg
    if DUMP:
        assert False, f"test_tax_revenue DUMP output: {emsg}"
    else:
        if emsg:
            reltol = RELTOL_ITAX.get(year, DEFAULT_RELTOL_ITAX)
            emsg += f"\nRELTOL_ITAX= {reltol:5.3f}"
            reltol = RELTOL_PTAX.get(year, DEFAULT_RELTOL_PTAX)
            emsg += f"\nRELTOL_PTAX= {reltol:5.3f}"
            raise ValueError(emsg)
