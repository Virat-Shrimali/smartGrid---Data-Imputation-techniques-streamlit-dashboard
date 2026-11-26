"""Synthetic MCAR (Missing Completely At Random) missing-value generation.

Provides :class:`MissingGenerator`, which randomly removes a specified
percentage of values from a numeric series, mirroring the reference paper's
1%, 3%, and 5% baseline rates and its 10-point stress increments through 90%,
followed by the 99% Makima stress test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import InvalidMissingRateError, get_logger

logger = get_logger(__name__)


@dataclass
class MissingGenerator:
    """Generates MCAR missing values on a numeric pandas Series.

    Attributes
    ----------
    random_state:
        Seed for the random number generator, ensuring reproducibility
        across repeated runs.
    """

    random_state: int = 42

    def generate(self, series: pd.Series, missing_pct: float) -> tuple[pd.Series, pd.Series, np.ndarray]:
        """Randomly remove ``missing_pct`` fraction of values (MCAR).

        Parameters
        ----------
        series:
            The complete, original numeric series (no NaNs assumed, but
            any existing NaNs are treated as already-missing and are not
            double-counted).
        missing_pct:
            Fraction of the *originally valid* values to remove, in the
            open interval (0, 1). For example, 0.05 removes 5% of the data.

        Returns
        -------
        tuple
            ``(series_with_missing, ground_truth, removed_indices)`` where
            ``series_with_missing`` is a copy of ``series`` with newly
            inserted NaNs, ``ground_truth`` is the original series
            (unchanged), and ``removed_indices`` is an array of integer
            positions that were set to NaN.

        Raises
        ------
        InvalidMissingRateError
            If ``missing_pct`` is not strictly between 0 and 1, or if the
            series does not contain enough valid points to remove the
            requested percentage.
        """
        if not (0.0 < missing_pct < 1.0):
            raise InvalidMissingRateError(
                f"missing_pct must be in the open interval (0, 1); got {missing_pct}."
            )

        ground_truth = series.copy()
        valid_positions = np.flatnonzero(series.notna().to_numpy())
        n_valid = len(valid_positions)
        n_to_remove = int(round(n_valid * missing_pct))

        if n_to_remove == 0:
            raise InvalidMissingRateError(
                f"missing_pct={missing_pct} results in 0 records removed out of "
                f"{n_valid} valid points. Increase the percentage or provide more data."
            )
        if n_to_remove >= n_valid:
            raise InvalidMissingRateError(
                f"missing_pct={missing_pct} would remove all {n_valid} valid points."
            )

        rng = np.random.default_rng(self.random_state)
        removed_indices = rng.choice(valid_positions, size=n_to_remove, replace=False)
        removed_indices.sort()

        series_with_missing = series.copy()
        series_with_missing.iloc[removed_indices] = np.nan

        logger.info(
            "Generated MCAR missingness: %.1f%% -> removed %d/%d valid points.",
            missing_pct * 100,
            n_to_remove,
            n_valid,
        )
        return series_with_missing, ground_truth, removed_indices

    def generate_multiple(
        self, series: pd.Series, missing_rates: tuple[float, ...]
    ) -> dict[float, tuple[pd.Series, pd.Series, np.ndarray]]:
        """Generate missing-data versions of a series across several rates.

        Parameters
        ----------
        series:
            The complete numeric series.
        missing_rates:
            An iterable of fractions (e.g. ``(0.01, 0.03, 0.05, 0.10, 0.20)``).

        Returns
        -------
        dict
            Mapping from missing rate to the tuple returned by
            :meth:`generate`.
        """
        return {rate: self.generate(series, rate) for rate in missing_rates}
