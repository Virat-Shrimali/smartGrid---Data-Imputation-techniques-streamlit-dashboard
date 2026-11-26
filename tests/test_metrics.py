"""Unit tests for src.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import MetricsCalculator
from src.utils import InsufficientDataError


def test_mae_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert MetricsCalculator.mae(a, a) == pytest.approx(0.0)


def test_mae_known_value():
    imputed = np.array([1.0, 2.0, 3.0])
    actual = np.array([2.0, 2.0, 5.0])
    # |1-2|+|2-2|+|3-5| = 1+0+2 = 3 -> mean = 1.0
    assert MetricsCalculator.mae(imputed, actual) == pytest.approx(1.0)


def test_rmse_known_value():
    imputed = np.array([0.0, 0.0])
    actual = np.array([3.0, 4.0])
    # sqrt((9+16)/2) = sqrt(12.5)
    assert MetricsCalculator.rmse(imputed, actual) == pytest.approx(np.sqrt(12.5))


def test_relative_error_percentage_scale():
    imputed = np.array([110.0])
    actual = np.array([100.0])
    assert MetricsCalculator.relative_error(imputed, actual) == pytest.approx(10.0)


def test_mape_percentage_scale():
    imputed = np.array([90.0])
    actual = np.array([100.0])
    assert MetricsCalculator.mape(imputed, actual) == pytest.approx(10.0)


def test_mismatched_lengths_raise():
    with pytest.raises(InsufficientDataError):
        MetricsCalculator.mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_empty_arrays_raise():
    with pytest.raises(InsufficientDataError):
        MetricsCalculator.mae(np.array([]), np.array([]))


def test_evaluate_returns_correct_structure():
    ground_truth = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    imputed = pd.Series([1.0, 2.5, 3.0, 3.5, 5.0])
    removed_indices = np.array([1, 3])

    result = MetricsCalculator.evaluate(
        technique="TestTech",
        missing_pct=0.4,
        imputed_series=imputed,
        ground_truth=ground_truth,
        removed_indices=removed_indices,
        execution_time_seconds=0.01,
    )

    assert result.technique == "TestTech"
    assert result.missing_pct == 0.4
    assert result.n_points == 2
    assert result.mae >= 0.0


def test_results_to_dataframe_columns():
    ground_truth = pd.Series([1.0, 2.0, 3.0])
    imputed = pd.Series([1.0, 2.1, 3.0])
    result = MetricsCalculator.evaluate(
        "Tech", 0.1, imputed, ground_truth, np.array([1]), 0.001
    )
    df = MetricsCalculator.results_to_dataframe([result])
    expected_cols = {
        "technique",
        "missing_pct",
        "mae",
        "rmse",
        "relative_error",
        "mape",
        "execution_time_seconds",
        "n_points",
        "n_runs",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1
