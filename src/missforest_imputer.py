"""MissForest imputation (Stekhoven & Bühlmann, 2012).

If the third-party ``missforest`` package is installed and importable, it
is used directly against the same lag-window feature representation used
by :class:`~src.knn_imputer.KNNImputer`. Otherwise, this module falls
back to a self-contained iterative Random-Forest imputer that reproduces
the core MissForest algorithm: missing values are initialized with the
column median, then repeatedly re-estimated with a
``RandomForestRegressor`` trained on the other feature columns, iterating
(in ascending order of missingness) until the imputed values stabilize
or ``max_iter`` is reached.

Only numeric data is supported, matching the rest of this package (which
operates on a single selected numeric column) — the categorical-variable
branch of the original MissForest algorithm (which would use a
``RandomForestClassifier``) therefore does not apply here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.imputer_base import BaseImputer
from src.utils import ImputationError, InsufficientDataError, get_logger

logger = get_logger(__name__)

try:
    from missforest import MissForest as _ExternalMissForest

    _EXTERNAL_MISSFOREST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional
    # third-party package (and its own dependencies) are installed.
    _ExternalMissForest = None
    _EXTERNAL_MISSFOREST_AVAILABLE = False


@dataclass
class MissForestImputer(BaseImputer):
    """Iterative Random-Forest imputer (MissForest-style).

    Attributes
    ----------
    n_estimators:
        Number of trees per ``RandomForestRegressor`` (default 100).
    max_depth:
        Maximum tree depth; ``None`` means unlimited (default None).
    max_iter:
        Maximum number of iterative refinement passes (default 5).
    random_state:
        Seed for reproducibility (default 42).
    n_lags:
        Number of lag/lead columns built on each side of the target value,
        reusing the same feature-engineering approach as
        :class:`~src.knn_imputer.KNNImputer` so a univariate series can be
        treated as a small multivariate table (default 4).
    tol:
        Relative-change threshold used for early stopping between
        iterations of the fallback algorithm (default 1e-3).
    """

    n_estimators: int = 50
    max_depth: int | None = None
    max_iter: int = 3
    random_state: int = 42
    n_lags: int = 4
    tol: float = 1e-3
    name: str = "MissForest"

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

    def _impute_with_external(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Delegate imputation to the third-party ``missforest`` package."""
        columns = [f"f{i}" for i in range(feature_matrix.shape[1])]
        frame = pd.DataFrame(feature_matrix, columns=columns)
        model = _ExternalMissForest()
        imputed_frame = model.fit_transform(frame)
        return imputed_frame.to_numpy(dtype=float)

    def _impute_with_fallback(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Self-contained iterative Random-Forest imputation (MissForest-style)."""
        matrix = feature_matrix.copy()
        n_rows, n_cols = matrix.shape
        missing_mask = np.isnan(matrix)

        if not missing_mask.any():
            return matrix

        # Step 1: initialize every missing cell with its column median
        # (falling back to the global median for an entirely-missing column).
        col_medians = np.nanmedian(matrix, axis=0)
        global_median = np.nanmedian(matrix)
        col_medians = np.where(np.isnan(col_medians), global_median, col_medians)
        for col in range(n_cols):
            col_mask = missing_mask[:, col]
            if col_mask.any():
                matrix[col_mask, col] = col_medians[col]

        # Step 2: iterate over columns in ascending order of missingness,
        # as in the original MissForest algorithm, re-estimating each
        # column's missing cells from a RandomForestRegressor trained on
        # the other columns.
        missing_counts = missing_mask.sum(axis=0)
        column_order = np.argsort(missing_counts)
        columns_with_missing = [int(c) for c in column_order if missing_counts[c] > 0]

        if not columns_with_missing:
            return matrix

        previous_matrix = matrix.copy()

        for iteration in range(self.max_iter):
            for col in columns_with_missing:
                row_mask = missing_mask[:, col]
                if not row_mask.any():
                    continue

                feature_cols = [c for c in range(n_cols) if c != col]
                X_train = matrix[~row_mask][:, feature_cols]
                y_train = matrix[~row_mask][:, col]
                X_predict = matrix[row_mask][:, feature_cols]

                if len(y_train) < 2:
                    continue  # not enough observed rows to fit a forest for this column

                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                matrix[row_mask, col] = model.predict(X_predict)

            delta = float(np.linalg.norm(matrix - previous_matrix))
            denom = float(np.linalg.norm(matrix)) + 1e-12
            relative_change = delta / denom
            logger.info(
                "[%s] Iteration %d/%d — relative change: %.6f",
                self.name,
                iteration + 1,
                self.max_iter,
                relative_change,
            )
            if relative_change < self.tol:
                logger.info("[%s] Converged after %d iteration(s).", self.name, iteration + 1)
                break
            previous_matrix = matrix.copy()

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
            if _EXTERNAL_MISSFOREST_AVAILABLE:
                logger.info("[%s] Using external 'missforest' package.", self.name)
                imputed_matrix = self._impute_with_external(feature_matrix)
            else:
                imputed_matrix = self._impute_with_fallback(feature_matrix)
        except Exception as exc:  # noqa: BLE001
            raise ImputationError(f"[{self.name}] Imputation failed: {exc}") from exc

        imputed_center = imputed_matrix[:, center_col]

        result = values.copy()
        missing_mask = np.isnan(values)
        result[missing_mask] = imputed_center[missing_mask]

        still_missing = np.isnan(result)
        if still_missing.any():
            fallback_value = np.nanmedian(values)
            result[still_missing] = fallback_value

        return pd.Series(result, index=series.index, name=series.name)
