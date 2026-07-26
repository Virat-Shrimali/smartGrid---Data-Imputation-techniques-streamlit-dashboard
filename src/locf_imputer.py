"""Last Observation Carried Forward (LOCF) imputation.

Replaces each missing value with the most recent valid value preceding
it. As documented in the reference paper, this technique fails when the
very first record is missing (there is no prior observation to carry
forward). This implementation handles that edge case gracefully by
falling back to the next valid observation (NOCB) for any leading gap,
while logging a warning, rather than crashing outright.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.imputer_base import BaseImputer
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LOCFImputer(BaseImputer):
    """Forward-fill imputer with a documented leading-NaN fallback."""

    name: str = "LOCF"

    def _impute(self, series: pd.Series) -> pd.Series:
        if pd.isna(series.iloc[0]):
            logger.warning(
                "[%s] First record is missing; no prior value exists to carry "
                "forward. Falling back to next-observation-carried-backward "
                "for the leading gap.",
                self.name,
            )

        # Forward fill handles the core LOCF behaviour.
        filled = series.ffill()
        # Any values still NaN must be a leading gap; back-fill as fallback.
        filled = filled.bfill()

        return filled
