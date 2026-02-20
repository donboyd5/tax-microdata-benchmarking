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
    expect = {"mean": 815.5521277934885, "sdev": 961.7270821801824}
    diffs = []
    for stat in ["mean", "sdev"]:
        act = actual[stat]
        exp = expect[stat]
        if not np.allclose([act], [exp]):
            diff = f"WEIGHT_DIFF:{stat},act,exp= {act} {exp}"
            diffs.append(diff)
    if diffs:
        emsg = "\nWEIGHT VARIABLE ACT-vs-EXP DIFFS:"
        for line in diffs:
            emsg += "\n" + line
        raise ValueError(emsg)
