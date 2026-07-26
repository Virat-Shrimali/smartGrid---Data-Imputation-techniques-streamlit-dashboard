"""Windowed Median Imputation.

Reproduces the MATLAB ``fillmissing(..., 'movmedian', k)`` behaviour used
in the reference paper: each missing value is replaced by the median of a
sliding window of ``window_size`` records centered on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.imputer_base import BaseImputer
from src.utils import ImputationError


@dataclass
class MedianImputer(BaseImputer):
    """Sliding-window median imputer.

    Attributes
    ----------
    window_size:
        Number of records in the centered moving window (default 10,
        matching the paper's configuration).
    """

    window_size: int = 10
    name: str = "Median"

    def _impute(self, series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float)
        n = len(values)
        missing_mask = np.isnan(values)

        if not missing_mask.any():
            return series

        half_window = self.window_size // 2
        result = values.copy()

        for idx in np.flatnonzero(missing_mask):
            lo = max(0, idx - half_window)
            hi = min(n, idx + half_window + 1)
            window = values[lo:hi]
            window_valid = window[~np.isnan(window)]

            if window_valid.size == 0:
                # Fall back to progressively wider windows before failing,
                # mirroring a robust production implementation rather than
                # crashing outright as the original MATLAB routine does.
                expanded_lo, expanded_hi = lo, hi
                while window_valid.size == 0 and (expanded_lo > 0 or expanded_hi < n):
                    expanded_lo = max(0, expanded_lo - self.window_size)
                    expanded_hi = min(n, expanded_hi + self.window_size)
                    window = values[expanded_lo:expanded_hi]
                    window_valid = window[~np.isnan(window)]

                if window_valid.size == 0:
                    raise ImputationError(
                        f"[{self.name}] No valid values found near index {idx} "
                        "even after expanding the window; series may be entirely missing."
                    )

            result[idx] = float(np.median(window_valid))

        return pd.Series(result, index=series.index, name=series.name)
