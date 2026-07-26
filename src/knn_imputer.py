"""K-Nearest Neighbor (KNN) imputation for a univariate time series.

scikit-learn's :class:`~sklearn.impute.KNNImputer` operates across
*columns* of a 2D feature matrix — it needs multiple correlated features
to compute meaningful distances. Since this package targets a single
selected time-series column, we engineer a lag-window feature matrix
(each row = a value plus its ``n_lags`` neighbours on either side),
apply ``KNNImputer`` on that matrix, and reconstruct the imputed values
from the center column. This preserves the spirit of the paper's
cross-column KNN approach (Euclidean distance over rows with no NaNs)
while remaining applicable to a single measurement column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer as SklearnKNNImputer

from src.imputer_base import BaseImputer
from src.utils import ImputationError, InsufficientDataError


@dataclass
class KNNImputer(BaseImputer):
    """KNN imputer based on a sliding lag-window feature representation.

    Attributes
    ----------
    n_neighbors:
        Number of neighbouring rows used to estimate each missing value.
    n_lags:
        Number of lag/lead columns built on each side of the target value.
        A larger window gives KNN more context to compute distances.
    """

    n_neighbors: int = 5
    n_lags: int = 4
    name: str = "KNN"

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
        n = len(values)

        min_required = self.n_neighbors + 1
        n_valid = int(np.sum(~np.isnan(values)))
        if n_valid < min_required:
            raise InsufficientDataError(
                f"[{self.name}] Need at least {min_required} valid points, found {n_valid}."
            )

        feature_matrix = self._build_feature_matrix(values)
        center_col = self.n_lags

        try:
            imputer = SklearnKNNImputer(n_neighbors=self.n_neighbors, weights="distance")
            imputed_matrix = imputer.fit_transform(feature_matrix)
        except ValueError as exc:
            raise ImputationError(f"[{self.name}] scikit-learn KNNImputer failed: {exc}") from exc

        imputed_center = imputed_matrix[:, center_col]

        result = values.copy()
        missing_mask = np.isnan(values)
        result[missing_mask] = imputed_center[missing_mask]

        # Extremely rare fallback: if KNN could not resolve a value (e.g. all
        # neighbouring lags were also NaN near the series boundary), use the
        # column-wide median as a last resort so the contract (no NaNs) holds.
        still_missing = np.isnan(result)
        if still_missing.any():
            fallback_value = np.nanmedian(values)
            result[still_missing] = fallback_value

        return pd.Series(result, index=series.index, name=series.name)
