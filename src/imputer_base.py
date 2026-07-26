"""Abstract base class shared by all imputation algorithms.

Every concrete imputer (Median, LOCF, KNN, Makima) inherits from
:class:`BaseImputer`, which defines a common interface, handles timing
instrumentation, and centralizes basic input validation so that
subclasses only need to implement the ``_impute`` method.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from src.utils import ImputationError, get_logger

logger = get_logger(__name__)


@dataclass
class ImputationResult:
    """Container for the outcome of a single imputation run.

    Attributes
    ----------
    imputed_series:
        The series after imputation (no remaining NaNs on success).
    technique_name:
        Human-readable name of the imputation technique.
    execution_time_seconds:
        Wall-clock time taken to run the imputation.
    n_imputed:
        Number of values that were filled in.
    """

    imputed_series: pd.Series
    technique_name: str
    execution_time_seconds: float
    n_imputed: int = field(default=0)


class BaseImputer(ABC):
    """Common interface and instrumentation for imputation algorithms.

    Subclasses must implement :meth:`_impute`, which receives a pandas
    Series containing NaNs and must return a fully imputed series of the
    same length and index.
    """

    #: Human-readable technique name, overridden by subclasses.
    name: str = "BaseImputer"

    def impute(self, series: pd.Series) -> ImputationResult:
        """Run the imputation algorithm with timing and validation.

        Parameters
        ----------
        series:
            A numeric pandas Series possibly containing NaN values.

        Returns
        -------
        ImputationResult
            The imputed series bundled with timing and metadata.

        Raises
        ------
        ImputationError
            If the underlying algorithm fails, or if it returns a result
            that still contains missing values or has a mismatched shape.
        """
        if series.empty:
            raise ImputationError(f"[{self.name}] Cannot impute an empty series.")

        n_missing_before = int(series.isna().sum())

        start = time.perf_counter()
        try:
            imputed = self._impute(series.copy())
        except ImputationError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap as domain-specific error
            raise ImputationError(f"[{self.name}] Imputation failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        if len(imputed) != len(series):
            raise ImputationError(
                f"[{self.name}] Output length {len(imputed)} does not match "
                f"input length {len(series)}."
            )

        remaining_nan = int(imputed.isna().sum())
        if remaining_nan > 0:
            raise ImputationError(
                f"[{self.name}] {remaining_nan} values remain missing after imputation."
            )

        logger.info(
            "%s imputed %d values in %.4f seconds.", self.name, n_missing_before, elapsed
        )

        return ImputationResult(
            imputed_series=imputed,
            technique_name=self.name,
            execution_time_seconds=elapsed,
            n_imputed=n_missing_before,
        )

    @abstractmethod
    def _impute(self, series: pd.Series) -> pd.Series:
        """Perform the actual imputation. Implemented by subclasses.

        Parameters
        ----------
        series:
            A copy of the input series (safe to mutate).

        Returns
        -------
        pandas.Series
            A series of the same length/index with all NaNs filled.
        """
        raise NotImplementedError
