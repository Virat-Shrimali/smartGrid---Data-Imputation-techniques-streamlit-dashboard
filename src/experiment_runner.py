"""Orchestration of the full imputation experiment pipeline.

:class:`ExperimentRunner` ties together data loading, missing-value
generation, all supported imputation algorithms, metric computation, and
result persistence into a single reproducible workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data_loader import DataLoader
from src.imputer_base import BaseImputer
from src.knn_imputer import KNNImputer
from src.locf_imputer import LOCFImputer
from src.makima_imputer import MakimaImputer
from src.median_imputer import MedianImputer
from src.mice_imputer import MICEImputer
from src.missforest_imputer import MissForestImputer
from src.metrics import MetricsCalculator, MetricsResult
from src.missing_generator import MissingGenerator
from src.pchip_imputer import PCHIPImputer
from src.utils import (
    DEFAULT_MISSING_RATES,
    ImputationError,
    InvalidMissingRateError,
    ensure_directory,
    get_logger,
)
from src.visualizer import Visualizer

logger = get_logger(__name__)


def default_imputers() -> list[BaseImputer]:
    """Return one fresh instance of each of the seven supported imputers."""
    return [
        MedianImputer(),
        LOCFImputer(),
        KNNImputer(),
        MakimaImputer(),
        PCHIPImputer(),
        MICEImputer(),
        MissForestImputer(),
    ]


#: Registry mapping technique names to their imputer classes, used by the
#: Streamlit UI to let users select algorithms by name.
IMPUTER_REGISTRY: dict[str, type[BaseImputer]] = {
    "Median": MedianImputer,
    "LOCF": LOCFImputer,
    "KNN": KNNImputer,
    "Makima": MakimaImputer,
    "PCHIP": PCHIPImputer,
    "MICE": MICEImputer,
    "MissForest": MissForestImputer,
}


def get_imputer(name: str) -> BaseImputer:
    """Instantiate an imputer by its registered technique name.

    Parameters
    ----------
    name:
        One of ``"Median"``, ``"LOCF"``, ``"KNN"``, ``"Makima"``.

    Returns
    -------
    BaseImputer
        A fresh instance of the requested imputer.

    Raises
    ------
    KeyError
        If ``name`` is not a registered technique.
    """
    if name not in IMPUTER_REGISTRY:
        raise KeyError(f"Unknown imputer '{name}'. Available: {list(IMPUTER_REGISTRY)}")
    return IMPUTER_REGISTRY[name]()


@dataclass
class ExperimentRunner:
    """Runs the end-to-end imputation benchmark for one dataset/column.

    Attributes
    ----------
    data_path:
        Path to the input CSV dataset.
    target_column:
        Name of the numeric column to run experiments on.
    results_dir:
        Directory where ``results.csv`` will be written.
    figures_dir:
        Directory where plots will be saved.
    missing_rates:
        Missing-data fractions to evaluate (default: 1/3/5/10/20/99%).
    imputers:
        List of imputer instances to benchmark (defaults to all seven).
    random_state:
        Seed used for reproducible MCAR deletion.
    """

    data_path: Path
    target_column: str
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")
    missing_rates: tuple[float, ...] = DEFAULT_MISSING_RATES
    imputers: list[BaseImputer] = field(default_factory=default_imputers)
    random_state: int = 42

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.results_dir = ensure_directory(Path(self.results_dir))
        self.figures_dir = ensure_directory(Path(self.figures_dir))
        self._loader = DataLoader(self.data_path)
        self._generator = MissingGenerator(random_state=self.random_state)
        self._visualizer = Visualizer(self.figures_dir)

    def run(self, generate_plots: bool = True) -> pd.DataFrame:
        """Execute the full experiment: load, generate gaps, impute, evaluate.

        Parameters
        ----------
        generate_plots:
            If True, saves the four standard plot types to ``figures_dir``.

        Returns
        -------
        pandas.DataFrame
            Long-form results table (one row per technique x missing_pct),
            also written to ``results_dir / 'results.csv'``.
        """
        self._loader.load()
        ground_truth = self._loader.select_column(self.target_column)

        all_results: list[MetricsResult] = []
        example_run: dict | None = None

        for missing_pct in self.missing_rates:
            try:
                series_with_missing, truth, removed_indices = self._generator.generate(
                    ground_truth, missing_pct
                )
            except InvalidMissingRateError as exc:
                logger.warning(
                    "Skipping %.0f%% missingness because it cannot be generated: %s",
                    missing_pct * 100,
                    exc,
                )
                continue

            for imputer in self.imputers:
                try:
                    outcome = imputer.impute(series_with_missing)
                except ImputationError as exc:
                    logger.error(
                        "Skipping %s at %.0f%% missing due to error: %s",
                        imputer.name,
                        missing_pct * 100,
                        exc,
                    )
                    continue

                metrics_result = MetricsCalculator.evaluate(
                    technique=outcome.technique_name,
                    missing_pct=missing_pct,
                    imputed_series=outcome.imputed_series,
                    ground_truth=truth,
                    removed_indices=removed_indices,
                    execution_time_seconds=outcome.execution_time_seconds,
                )
                all_results.append(metrics_result)

                if generate_plots and (example_run is None or missing_pct == self.missing_rates[0]):
                    self._visualizer.plot_original_vs_imputed(
                        ground_truth=truth,
                        imputed_series=outcome.imputed_series,
                        removed_indices=removed_indices,
                        technique=outcome.technique_name,
                        missing_pct=missing_pct,
                    )

        results_df = MetricsCalculator.results_to_dataframe(all_results)
        results_path = self.results_dir / "results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info("Saved results table to %s (%d rows).", results_path, len(results_df))

        if generate_plots and not results_df.empty:
            self._visualizer.plot_error_vs_missing_pct(results_df, metric="relative_error")
            self._visualizer.plot_runtime_comparison(results_df)
            self._visualizer.plot_metric_comparison(results_df)

        return results_df
