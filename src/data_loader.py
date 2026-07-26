"""CSV time-series data loading and validation.

Provides :class:`DataLoader`, responsible for reading an arbitrary CSV
time-series dataset, validating its structure, and exposing a single
numeric column selected by the caller for downstream imputation
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils import (
    ColumnNotFoundError,
    DataLoadError,
    InsufficientDataError,
    NonNumericColumnError,
    get_logger,
)

logger = get_logger(__name__)


@dataclass
class DataLoader:
    """Loads a CSV time-series dataset and selects a numeric target column.

    Attributes
    ----------
    filepath:
        Path to the CSV file on disk.
    dataframe:
        The loaded pandas DataFrame (populated after calling :meth:`load`).
    """

    filepath: Path
    dataframe: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)

    def load(self) -> pd.DataFrame:
        """Read the CSV file from disk into memory.

        Returns
        -------
        pandas.DataFrame
            The loaded dataset.

        Raises
        ------
        DataLoadError
            If the file does not exist, is empty, or cannot be parsed.
        """
        if not self.filepath.exists():
            raise DataLoadError(f"Dataset not found at path: {self.filepath}")

        if self.filepath.suffix.lower() != ".csv":
            raise DataLoadError(
                f"Unsupported file extension '{self.filepath.suffix}'. Only .csv is supported."
            )

        try:
            self.dataframe = pd.read_csv(self.filepath)
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"The CSV file at {self.filepath} is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"Failed to parse CSV file at {self.filepath}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain-specific error
            raise DataLoadError(f"Unexpected error while loading {self.filepath}: {exc}") from exc

        if self.dataframe.empty:
            raise DataLoadError(f"Dataset at {self.filepath} contains no rows.")

        logger.info(
            "Loaded dataset '%s' with shape %s.", self.filepath.name, self.dataframe.shape
        )
        return self.dataframe

    def list_numeric_columns(self) -> list[str]:
        """Return the names of all numeric columns in the loaded dataset.

        Returns
        -------
        list[str]
            Numeric column names.

        Raises
        ------
        DataLoadError
            If :meth:`load` has not been called yet.
        """
        if self.dataframe is None:
            raise DataLoadError("Dataset has not been loaded yet. Call load() first.")
        return self.dataframe.select_dtypes(include="number").columns.tolist()

    def select_column(self, column: str, min_valid_points: int = 30) -> pd.Series:
        """Select and validate a single numeric column for imputation.

        Parameters
        ----------
        column:
            Name of the column to select.
        min_valid_points:
            Minimum number of non-missing values required for the column
            to be usable in downstream experiments.

        Returns
        -------
        pandas.Series
            The selected column as a float series, indexed identically to
            the source dataframe.

        Raises
        ------
        DataLoadError
            If the dataset has not been loaded yet.
        ColumnNotFoundError
            If ``column`` does not exist in the dataset.
        NonNumericColumnError
            If ``column`` exists but is not numeric.
        InsufficientDataError
            If the column has fewer than ``min_valid_points`` valid values.
        """
        if self.dataframe is None:
            raise DataLoadError("Dataset has not been loaded yet. Call load() first.")

        if column not in self.dataframe.columns:
            raise ColumnNotFoundError(
                f"Column '{column}' not found. Available columns: "
                f"{list(self.dataframe.columns)}"
            )

        series = self.dataframe[column]

        if not pd.api.types.is_numeric_dtype(series):
            raise NonNumericColumnError(
                f"Column '{column}' is not numeric (dtype={series.dtype})."
            )

        series = series.astype(float)
        valid_count = int(series.notna().sum())

        if valid_count < min_valid_points:
            raise InsufficientDataError(
                f"Column '{column}' has only {valid_count} valid points; "
                f"at least {min_valid_points} are required."
            )

        logger.info(
            "Selected column '%s' (%d valid / %d total points).",
            column,
            valid_count,
            len(series),
        )
        return series

    @staticmethod
    def create_example_dataset(path: Path, n_points: int = 2000, seed: int = 7) -> Path:
        """Generate and save a synthetic smart-grid-style CSV dataset.

        Useful for demos and for the Streamlit app's "load example dataset"
        action when the user has no CSV of their own to upload.

        Parameters
        ----------
        path:
            Destination path for the generated CSV file.
        n_points:
            Number of time-series rows to generate.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        Path
            The path the dataset was written to.
        """
        import numpy as np

        path = Path(path)
        rng = np.random.default_rng(seed)
        t = np.arange(n_points)

        voltage_a = 220 + 3 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.8, n_points)
        voltage_b = 219 + 3 * np.sin(2 * np.pi * t / 96 + 2.09) + rng.normal(0, 0.8, n_points)
        voltage_c = 221 + 3 * np.sin(2 * np.pi * t / 96 + 4.19) + rng.normal(0, 0.8, n_points)
        current_a = 15 + 8 * np.sin(2 * np.pi * t / 96 + 0.5) + rng.normal(0, 1.2, n_points)
        current_b = 14 + 9 * np.sin(2 * np.pi * t / 96 + 0.9) + rng.normal(0, 1.4, n_points)
        current_c = 16 + 7 * np.sin(2 * np.pi * t / 96 + 1.3) + rng.normal(0, 1.1, n_points)
        room_temp = 25 + 5 * np.sin(2 * np.pi * t / (96 * 30)) + rng.normal(0, 0.3, n_points)
        transformer_temp = 40 + 8 * np.sin(2 * np.pi * t / (96 * 30) + 0.4) + rng.normal(0, 0.5, n_points)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2019-06-09", periods=n_points, freq="min"),
                "secondary_voltage_a": voltage_a,
                "secondary_voltage_b": voltage_b,
                "secondary_voltage_c": voltage_c,
                "primary_current_a": current_a,
                "primary_current_b": current_b,
                "primary_current_c": current_c,
                "room_temperature": room_temp,
                "transformer_temperature": transformer_temp,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Generated example dataset at %s (%d rows).", path, len(df))
        return path

    def summary(self) -> pd.DataFrame:
        """Return descriptive statistics and missingness info for all columns.

        Returns
        -------
        pandas.DataFrame
            A summary table with dtype, missing count, missing percentage,
            and basic statistics for numeric columns.
        """
        if self.dataframe is None:
            raise DataLoadError("Dataset has not been loaded yet. Call load() first.")

        n_rows = len(self.dataframe)
        rows = []
        for col in self.dataframe.columns:
            series = self.dataframe[col]
            missing = int(series.isna().sum())
            rows.append(
                {
                    "column": col,
                    "dtype": str(series.dtype),
                    "missing_count": missing,
                    "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
                    "is_numeric": pd.api.types.is_numeric_dtype(series),
                }
            )
        return pd.DataFrame(rows)
