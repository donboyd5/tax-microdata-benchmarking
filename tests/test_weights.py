"""
Test of tmd/storage/output/tmd.csv.gz weights.
"""

import numpy as np
import pytest


@pytest.mark.weight_distribution
def test_weights(tmd_variables):
    """
    Calculate distribution of TMD weight variable.
    """
    wght = tmd_variables["s006"].to_numpy()
    actual = {"mean": wght.mean(), "sdev": wght.std()}
    expect = {"mean": 816.06972, "sdev": 1140.202438}
    tolerance = {"mean": 0.0015, "sdev": 0.0005}
    diffs = []
    for stat in ["mean", "sdev"]:
        act = actual[stat]
        exp = expect[stat]
        abstol = 0.0
        reltol = tolerance[stat]
        if not np.allclose([act], [exp], atol=abstol, rtol=reltol):
            diff = (
                f"WEIGHT_DIFF:{stat},act,exp,atol,rtol= "
                f"{act} {exp} {abstol} {reltol}"
            )
            diffs.append(diff)
    if diffs:
        emsg = "\nWEIGHT VARIABLE ACT-vs-EXP DIFFS:"
        for line in diffs:
            emsg += "\n" + line
        raise ValueError(emsg)
