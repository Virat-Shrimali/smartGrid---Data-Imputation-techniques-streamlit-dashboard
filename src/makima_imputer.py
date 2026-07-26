"""Makima (Modified Akima) interpolation imputation.

Uses SciPy's ``Akima1DInterpolator`` with ``method='makima'`` (available
in SciPy >= 1.13), which reproduces MATLAB's ``fillmissing(..., 'makima')``
behaviour: piecewise cubic interpolation with continuous first-order
derivatives and suppressed oscillation. If an older SciPy version without
the ``method`` keyword is detected, falls back to the standard Akima
interpolator, which still avoids the overshoot typical of cubic splines.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

from src.imputer_base import BaseImputer
from src.utils import ImputationError, InsufficientDataError, get_logger

logger = get_logger(__name__)


@dataclass
class MakimaImputer(BaseImputer):
    """Piecewise-cubic Makima interpolation imputer."""

    name: str = "Makima"

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
            supports_makima = "method" in inspect.signature(Akima1DInterpolator.__init__).parameters
            if supports_makima:
                interpolator = Akima1DInterpolator(x_valid, y_valid, method="makima")
            else:
                logger.warning(
                    "[%s] Installed SciPy version does not support method='makima'; "
                    "falling back to standard Akima interpolation.",
                    self.name,
                )
                interpolator = Akima1DInterpolator(x_valid, y_valid)
        except Exception as exc:  # noqa: BLE001
            raise ImputationError(f"[{self.name}] Failed to build interpolator: {exc}") from exc

        result = values.copy()
        missing_idx = x[~valid_mask]

        # Akima1DInterpolator extrapolates as NaN outside [x_valid.min(), x_valid.max()]
        # by default; clip missing indices to the valid range and use edge values
        # for anything genuinely outside that range (leading/trailing gaps).
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
