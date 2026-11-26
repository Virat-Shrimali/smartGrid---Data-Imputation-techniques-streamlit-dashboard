"""Unit tests for all four imputation algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.knn_imputer import KNNImputer
from src.locf_imputer import LOCFImputer
from src.makima_imputer import MakimaImputer
from src.median_imputer import MedianImputer
from src.missing_generator import MissingGenerator
from src.utils import ImputationError


@pytest.fixture
def series_with_gaps() -> pd.Series:
    rng = np.random.default_rng(11)
    n = 200
    t = np.arange(n)
    values = 10 + 3 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.3, n)
    series = pd.Series(values)
    missing_positions = rng.choice(np.arange(5, n - 5), size=20, replace=False)
    series.iloc[missing_positions] = np.nan
    return series


@pytest.mark.parametrize(
    "imputer_cls",
    [MedianImputer, LOCFImputer, KNNImputer, MakimaImputer],
)
def test_imputer_fills_all_nans(imputer_cls, series_with_gaps):
    imputer = imputer_cls()
    result = imputer.impute(series_with_gaps)
    assert result.imputed_series.isna().sum() == 0
    assert len(result.imputed_series) == len(series_with_gaps)
    assert result.n_imputed == 20
    assert result.execution_time_seconds >= 0.0


@pytest.mark.parametrize(
    "imputer_cls",
    [MedianImputer, LOCFImputer, KNNImputer, MakimaImputer],
)
def test_imputer_preserves_known_values(imputer_cls, series_with_gaps):
    imputer = imputer_cls()
    result = imputer.impute(series_with_gaps)
    known_mask = series_with_gaps.notna()
    pd.testing.assert_series_equal(
        result.imputed_series[known_mask].reset_index(drop=True),
        series_with_gaps[known_mask].reset_index(drop=True),
        check_names=False,
    )


def test_median_imputer_uses_local_window():
    # A step function: with an asymmetric gap position, the local window
    # median should reflect the nearby neighbourhood, not a distant region.
    values = [1.0] * 20 + [np.nan] + [1.0] * 5 + [100.0] * 20
    series = pd.Series(values)
    imputer = MedianImputer(window_size=4)
    result = imputer.impute(series)
    imputed_value = result.imputed_series.iloc[20]
    # The window around index 20 only reaches nearby 1.0 values, not the
    # distant 100.0 plateau, so the imputed value should equal 1.0.
    assert imputed_value == pytest.approx(1.0)


def test_locf_handles_leading_nan_without_crashing():
    series = pd.Series([np.nan, np.nan, 5.0, 6.0, 7.0])
    imputer = LOCFImputer()
    result = imputer.impute(series)
    assert result.imputed_series.isna().sum() == 0
    # Leading NaNs should fall back to the next valid observation (5.0).
    assert result.imputed_series.iloc[0] == pytest.approx(5.0)


def test_makima_interpolates_smooth_trend():
    x = np.arange(50)
    y = np.sin(x / 5.0)
    series = pd.Series(y)
    series.iloc[[10, 11, 12]] = np.nan

    imputer = MakimaImputer()
    result = imputer.impute(series)
    # Interpolated values should be reasonably close to the true sine curve.
    true_vals = np.sin(np.array([10, 11, 12]) / 5.0)
    imputed_vals = result.imputed_series.iloc[[10, 11, 12]].to_numpy()
    np.testing.assert_allclose(imputed_vals, true_vals, atol=0.1)


def test_makima_completes_99_percent_missingness():
    original = pd.Series(np.sin(np.arange(500) / 30.0))
    with_missing, _, removed = MissingGenerator(random_state=1).generate(original, 0.99)

    result = MakimaImputer().impute(with_missing)

    assert len(removed) == 495
    assert result.imputed_series.notna().all()
    pd.testing.assert_series_equal(
        result.imputed_series[with_missing.notna()],
        original[with_missing.notna()],
    )


def test_empty_series_raises_for_all_imputers():
    empty = pd.Series([], dtype=float)
    for imputer_cls in (MedianImputer, LOCFImputer, KNNImputer, MakimaImputer):
        with pytest.raises(ImputationError):
            imputer_cls().impute(empty)


def test_makima_requires_at_least_two_valid_points():
    series = pd.Series([np.nan, np.nan, 5.0])
    with pytest.raises(ImputationError):
        MakimaImputer().impute(series)
