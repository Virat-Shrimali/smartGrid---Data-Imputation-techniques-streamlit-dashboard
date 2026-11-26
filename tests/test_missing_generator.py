"""Unit tests for src.missing_generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.missing_generator import MissingGenerator
from src.utils import InvalidMissingRateError


@pytest.fixture
def clean_series() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(size=500))


def test_generate_removes_correct_count(clean_series):
    gen = MissingGenerator(random_state=1)
    with_missing, ground_truth, removed = gen.generate(clean_series, 0.10)
    assert with_missing.isna().sum() == 50
    assert len(removed) == 50
    assert ground_truth.isna().sum() == 0  # ground truth untouched


def test_generate_supports_99_percent_missingness(clean_series):
    gen = MissingGenerator(random_state=1)
    with_missing, ground_truth, removed = gen.generate(clean_series, 0.99)

    assert len(removed) == 495
    assert with_missing.isna().sum() == 495
    assert with_missing.notna().sum() == 5
    pd.testing.assert_series_equal(ground_truth, clean_series)


def test_generate_is_reproducible(clean_series):
    gen1 = MissingGenerator(random_state=42)
    gen2 = MissingGenerator(random_state=42)
    _, _, removed1 = gen1.generate(clean_series, 0.05)
    _, _, removed2 = gen2.generate(clean_series, 0.05)
    np.testing.assert_array_equal(removed1, removed2)


def test_generate_different_seeds_differ(clean_series):
    gen1 = MissingGenerator(random_state=1)
    gen2 = MissingGenerator(random_state=2)
    _, _, removed1 = gen1.generate(clean_series, 0.05)
    _, _, removed2 = gen2.generate(clean_series, 0.05)
    assert not np.array_equal(removed1, removed2)


@pytest.mark.parametrize("bad_pct", [0.0, 1.0, -0.1, 1.5])
def test_invalid_missing_pct_raises(clean_series, bad_pct):
    gen = MissingGenerator()
    with pytest.raises(InvalidMissingRateError):
        gen.generate(clean_series, bad_pct)


def test_generate_multiple_returns_all_rates(clean_series):
    gen = MissingGenerator(random_state=3)
    rates = (0.01, 0.05, 0.10)
    results = gen.generate_multiple(clean_series, rates)
    assert set(results.keys()) == set(rates)
    for rate, (with_missing, _, removed) in results.items():
        expected = int(round(len(clean_series) * rate))
        assert len(removed) == expected


def test_original_values_preserved_at_non_removed_positions(clean_series):
    gen = MissingGenerator(random_state=5)
    with_missing, ground_truth, removed = gen.generate(clean_series, 0.10)
    kept_mask = np.ones(len(clean_series), dtype=bool)
    kept_mask[removed] = False
    pd.testing.assert_series_equal(
        with_missing[kept_mask].reset_index(drop=True),
        ground_truth[kept_mask].reset_index(drop=True),
    )
