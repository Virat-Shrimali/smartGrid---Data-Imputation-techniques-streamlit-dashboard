"""PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) imputation.

Uses SciPy's ``PchipInterpolator``, which builds a shape-preserving cubic
Hermite spline: unlike a plain cubic spline, PCHIP never overshoots and
preserves monotonicity between consecutive data points. This makes it a
useful complement to Makima for signals with sharp monotonic ramps or
plateaus where oscillation-free, monotone-preserving interpolation is
preferred over smoothness of higher derivatives.

Follows the same interface and edge-case handling as
:class:`~src.makima_imputer.MakimaImputer` (only numeric columns are
supported, matching the rest of this package).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.imputer_base import BaseImputer
from src.utils import ImputationError, InsufficientDataError


@dataclass
class PCHIPImputer(BaseImputer):
    """Shape-preserving piecewise cubic Hermite interpolation imputer."""

    name: str = "PCHIP"

    def _impute(self, series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float)
        n = len(values)
        x = np.arange(n)

        valid_mask = ~np.isnan(values)
        n_valid = int(valid_mask.sum())

        if n_valid < 2:
            raise InsufficientDataError(
                f"[{self.name}] Need at least 2 valid points to interpolate, found {n_valid}."
            )

        x_valid = x[valid_mask]
        y_valid = values[valid_mask]

        try:
            interpolator = PchipInterpolator(x_valid, y_valid, extrapolate=False)
        except Exception as exc:  # noqa: BLE001
            raise ImputationError(f"[{self.name}] Failed to build interpolator: {exc}") from exc

        result = values.copy()
        missing_idx = x[~valid_mask]

        # PchipInterpolator with extrapolate=False returns NaN outside
        # [x_valid.min(), x_valid.max()]; clip missing indices to the valid
        # range and use edge values for genuine leading/trailing gaps,
        # matching MakimaImputer's boundary-handling behaviour.
        in_range = (missing_idx >= x_valid.min()) & (missing_idx <= x_valid.max())

        if in_range.any():
            interpolated = interpolator(missing_idx[in_range])
            result[missing_idx[in_range]] = interpolated

        out_of_range = missing_idx[~in_range]
        if len(out_of_range) > 0:
            for idx in out_of_range:
                if idx < x_valid.min():
                    result[idx] = y_valid[0]
                else:
                    result[idx] = y_valid[-1]

        return pd.Series(result, index=series.index, name=series.name)
