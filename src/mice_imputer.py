"""MICE (Multiple Imputation by Chained Equations) imputation.

Uses scikit-learn's ``IterativeImputer`` — its implementation of the MICE
algorithm — with a ``BayesianRidge`` estimator. As with
:class:`~src.knn_imputer.KNNImputer`, ``IterativeImputer`` needs multiple
correlated columns to model each missing value as a function of the
others, so the univariate series is expanded into a lag-window feature
matrix (each row = a value plus ``n_lags`` neighbours on either side)
before imputation, and the center column is extracted afterward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from src.imputer_base import BaseImputer
from src.utils import ImputationError, InsufficientDataError


@dataclass
class MICEImputer(BaseImputer):
    """MICE imputer based on scikit-learn's IterativeImputer + BayesianRidge.

    Attributes
    ----------
    max_iter:
        Maximum number of chained-equation imputation rounds (default 10).
    random_state:
        Seed for reproducibility (default 42).
    tol:
        Convergence tolerance passed to ``IterativeImputer`` (default 1e-3).
    n_lags:
        Number of lag/lead columns built on each side of the target value,
        giving ``IterativeImputer`` multiple correlated features to
        condition on (default 4, matching :class:`~src.knn_imputer.KNNImputer`).
    """

    max_iter: int = 10
    random_state: int = 42
    tol: float = 1e-3
    n_lags: int = 4
    name: str = "MICE"

    def _build_feature_matrix(self, values: np.ndarray) -> np.ndarray:
        """Construct a (n_samples, 2*n_lags + 1) matrix of lagged features."""
        n = len(values)
        width = 2 * self.n_lags + 1
        matrix = np.full((n, width), np.nan, dtype=float)

        for offset in range(-self.n_lags, self.n_lags + 1):
            col_idx = offset + self.n_lags
            if offset < 0:
                matrix[-offset:, col_idx] = values[: n + offset]
            elif offset > 0:
                matrix[: n - offset, col_idx] = values[offset:]
            else:
                matrix[:, col_idx] = values

        return matrix

    def _impute(self, series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float)
        n_missing = int(np.isnan(values).sum())

        if n_missing == 0:
            return series

        n_valid = len(values) - n_missing
        if n_valid < 2:
            raise InsufficientDataError(
                f"[{self.name}] Need at least 2 valid points, found {n_valid}."
            )

        feature_matrix = self._build_feature_matrix(values)
        center_col = self.n_lags

        try:
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                sample_posterior=False,
            )
            imputed_matrix = imputer.fit_transform(feature_matrix)
        except ValueError as exc:
            raise ImputationError(
                f"[{self.name}] scikit-learn IterativeImputer failed: {exc}"
            ) from exc

        imputed_center = imputed_matrix[:, center_col]

        result = values.copy()
        missing_mask = np.isnan(values)
        result[missing_mask] = imputed_center[missing_mask]

        # Rare fallback for boundary positions IterativeImputer could not
        # resolve (e.g. all lag columns simultaneously missing).
        still_missing = np.isnan(result)
        if still_missing.any():
            fallback_value = np.nanmedian(values)
            result[still_missing] = fallback_value

        return pd.Series(result, index=series.index, name=series.name)
