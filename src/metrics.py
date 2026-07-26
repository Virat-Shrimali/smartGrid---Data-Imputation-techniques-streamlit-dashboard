"""Evaluation metrics for comparing imputed values against ground truth.

Implements MAE, RMSE, Relative Error (mean relative error, matching the
paper's ``ERmed``), and MAPE. Execution time is captured separately by
:class:`src.imputer_base.ImputationResult` and merged in here for
convenience when building comparison tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import InsufficientDataError, get_logger

logger = get_logger(__name__)

_EPSILON = 1e-10  # Guards against division by zero for near-zero ground-truth values.


@dataclass
class MetricsResult:
    """Bundle of evaluation metrics for one imputation run.

    Attributes
    ----------
    technique:
        Name of the imputation technique evaluated.
    missing_pct:
        The missing-data percentage used to generate the gaps evaluated.
    mae, rmse, relative_error, mape:
        Standard error metrics computed only over the positions that were
        actually imputed (i.e. originally removed).
    execution_time_seconds:
        Wall-clock runtime of the imputation algorithm.
    n_points:
        Number of imputed points the metrics were computed over.
    """

    technique: str
    missing_pct: float
    mae: float
    rmse: float
    relative_error: float
    mape: float
    execution_time_seconds: float
    n_points: int

    def to_dict(self) -> dict:
        """Convert this result into a flat dictionary (for CSV export)."""
        return {
            "technique": self.technique,
            "missing_pct": self.missing_pct,
            "mae": self.mae,
            "rmse": self.rmse,
            "relative_error": self.relative_error,
            "mape": self.mape,
            "execution_time_seconds": self.execution_time_seconds,
            "n_points": self.n_points,
        }


class MetricsCalculator:
    """Stateless collection of static metric computation helpers."""

    @staticmethod
    def _validate(imputed: np.ndarray, actual: np.ndarray) -> None:
        if len(imputed) != len(actual):
            raise InsufficientDataError(
                f"Imputed ({len(imputed)}) and actual ({len(actual)}) arrays have "
                "mismatched lengths."
            )
        if len(imputed) == 0:
            raise InsufficientDataError("Cannot compute metrics on empty arrays.")

    @staticmethod
    def mae(imputed: np.ndarray, actual: np.ndarray) -> float:
        """Mean Absolute Error."""
        MetricsCalculator._validate(imputed, actual)
        return float(np.mean(np.abs(imputed - actual)))

    @staticmethod
    def rmse(imputed: np.ndarray, actual: np.ndarray) -> float:
        """Root Mean Squared Error."""
        MetricsCalculator._validate(imputed, actual)
        return float(np.sqrt(np.mean((imputed - actual) ** 2)))

    @staticmethod
    def relative_error(imputed: np.ndarray, actual: np.ndarray) -> float:
        """Mean Relative Error, expressed as a percentage (matches paper's ERmed).

        Computed per-point as ``|imputed - actual| / |actual|`` (with a small
        epsilon guard against division by zero), then averaged and reported
        as a percentage.
        """
        MetricsCalculator._validate(imputed, actual)
        denom = np.where(np.abs(actual) < _EPSILON, _EPSILON, np.abs(actual))
        per_point = np.abs(imputed - actual) / denom
        return float(np.mean(per_point) * 100)

    @staticmethod
    def mape(imputed: np.ndarray, actual: np.ndarray) -> float:
        """Mean Absolute Percentage Error, as a percentage."""
        MetricsCalculator._validate(imputed, actual)
        denom = np.where(np.abs(actual) < _EPSILON, _EPSILON, np.abs(actual))
        return float(np.mean(np.abs((actual - imputed) / denom)) * 100)

    @classmethod
    def evaluate(
        cls,
        technique: str,
        missing_pct: float,
        imputed_series: pd.Series,
        ground_truth: pd.Series,
        removed_indices: np.ndarray,
        execution_time_seconds: float,
    ) -> MetricsResult:
        """Compute all metrics for the positions that were imputed.

        Parameters
        ----------
        technique:
            Name of the imputation technique.
        missing_pct:
            The missing-data fraction used for this run.
        imputed_series:
            Series after imputation (full length).
        ground_truth:
            Original, complete series (full length).
        removed_indices:
            Integer positions that were artificially set to NaN and then
            imputed; metrics are computed only over these positions.
        execution_time_seconds:
            Runtime of the imputation algorithm.

        Returns
        -------
        MetricsResult
        """
        imputed_vals = imputed_series.iloc[removed_indices].to_numpy(dtype=float)
        actual_vals = ground_truth.iloc[removed_indices].to_numpy(dtype=float)

        result = MetricsResult(
            technique=technique,
            missing_pct=missing_pct,
            mae=cls.mae(imputed_vals, actual_vals),
            rmse=cls.rmse(imputed_vals, actual_vals),
            relative_error=cls.relative_error(imputed_vals, actual_vals),
            mape=cls.mape(imputed_vals, actual_vals),
            execution_time_seconds=execution_time_seconds,
            n_points=len(removed_indices),
        )
        logger.info(
            "Metrics for %s @ %.1f%%: MAE=%.4f RMSE=%.4f RelErr=%.2f%% MAPE=%.2f%% (%.4fs)",
            technique,
            missing_pct * 100,
            result.mae,
            result.rmse,
            result.relative_error,
            result.mape,
            execution_time_seconds,
        )
        return result

    @staticmethod
    def results_to_dataframe(results: list[MetricsResult]) -> pd.DataFrame:
        """Convert a list of :class:`MetricsResult` into a tidy DataFrame."""
        if not results:
            return pd.DataFrame(
                columns=[
                    "technique",
                    "missing_pct",
                    "mae",
                    "rmse",
                    "relative_error",
                    "mape",
                    "execution_time_seconds",
                    "n_points",
                ]
            )
        return pd.DataFrame([r.to_dict() for r in results])
