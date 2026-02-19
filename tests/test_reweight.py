"""
Unit tests for tmd/utils/reweight.py helper functions.
"""

import numpy as np
import pandas as pd
import pytest
from tmd.utils.reweight import _drop_impossible_targets


def test_drop_impossible_targets_removes_all_zero_column():
    """All-zero columns are impossible targets and must be dropped."""
    loss_matrix = pd.DataFrame(
        {
            "good_a": [1.0, 2.0, 0.0],
            "bad_zero": [0.0, 0.0, 0.0],
            "good_b": [0.0, 3.0, 1.0],
        }
    )
    targets_arr = np.array([100.0, 50.0, 200.0])
    result_matrix, result_targets = _drop_impossible_targets(
        loss_matrix, targets_arr
    )
    assert "bad_zero" not in result_matrix.columns
    assert list(result_matrix.columns) == ["good_a", "good_b"]
    np.testing.assert_array_equal(result_targets, [100.0, 200.0])


def test_drop_impossible_targets_keeps_all_when_none_zero():
    """No columns are dropped when none are all-zero."""
    loss_matrix = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        }
    )
    targets_arr = np.array([10.0, 20.0])
    result_matrix, result_targets = _drop_impossible_targets(
        loss_matrix, targets_arr
    )
    assert list(result_matrix.columns) == ["a", "b"]
    np.testing.assert_array_equal(result_targets, [10.0, 20.0])


def test_drop_impossible_targets_column_with_single_nonzero_is_kept():
    """A column with at least one nonzero value is not impossible."""
    loss_matrix = pd.DataFrame(
        {
            "almost_zero": [0.0, 0.0, 1e-10],
        }
    )
    targets_arr = np.array([5.0])
    result_matrix, result_targets = _drop_impossible_targets(
        loss_matrix, targets_arr
    )
    assert "almost_zero" in result_matrix.columns
    assert len(result_targets) == 1
