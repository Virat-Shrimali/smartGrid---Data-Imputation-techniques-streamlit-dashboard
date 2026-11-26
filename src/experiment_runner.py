"""Orchestration of the full imputation experiment pipeline.

:class:`ExperimentRunner` ties together data loading, missing-value
generation, all supported imputation algorithms, metric computation, and
result persistence into a single reproducible workflow.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.imputer_base import BaseImputer, ImputationResult
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
    PAPER_RUN_REPEATS,
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


def run_repeated_imputation(
    imputer: BaseImputer,
    ground_truth: pd.Series,
    missing_pct: float,
    random_state: int = 42,
) -> tuple[MetricsResult, ImputationResult, np.ndarray]:
    """Run one imputer on eight independently generated MCAR masks.

    Returns the mean metrics across successful runs and the first imputed
    series for visualization. Each repetition uses the same seed sequence
    across algorithms so their results remain paired.
    """
    metrics_runs: list[MetricsResult] = []
    first_outcome: ImputationResult | None = None
    first_removed_indices: np.ndarray | None = None

    for repetition in range(PAPER_RUN_REPEATS):
        generator = MissingGenerator(random_state=random_state + repetition)
        series_with_missing, truth, removed_indices = generator.generate(
            ground_truth, missing_pct
        )
        try:
            outcome = deepcopy(imputer).impute(series_with_missing)
        except ImputationError as exc:
            logger.warning(
                "%s failed at %.0f%% missingness, repetition %d/%d: %s",
                imputer.name,
                missing_pct * 100,
                repetition + 1,
                PAPER_RUN_REPEATS,
                exc,
            )
            continue

        if first_outcome is None:
            first_outcome = outcome
            first_removed_indices = removed_indices
        metrics_runs.append(
            MetricsCalculator.evaluate(
                technique=outcome.technique_name,
                missing_pct=missing_pct,
                imputed_series=outcome.imputed_series,
                ground_truth=truth,
                removed_indices=removed_indices,
                execution_time_seconds=outcome.execution_time_seconds,
            )
        )

    if not metrics_runs or first_outcome is None or first_removed_indices is None:
        raise ImputationError(
            f"[{imputer.name}] All {PAPER_RUN_REPEATS} repetitions failed at "
            f"{missing_pct * 100:.0f}% missingness."
        )

    means = pd.DataFrame([result.to_dict() for result in metrics_runs]).mean(
        numeric_only=True
    )
    return (
        MetricsResult(
            technique=metrics_runs[0].technique,
            missing_pct=missing_pct,
            mae=float(means["mae"]),
            rmse=float(means["rmse"]),
            relative_error=float(means["relative_error"]),
            mape=float(means["mape"]),
            execution_time_seconds=float(means["execution_time_seconds"]),
            n_points=metrics_runs[0].n_points,
            n_runs=len(metrics_runs),
        ),
        first_outcome,
        first_removed_indices,
    )


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
        Missing-data fractions to evaluate (default: 1/3/5%, then 10-point
        increments from 10% through 90%, followed by 99%).
    imputers:
        List of imputer instances to benchmark (defaults to all seven).
    random_state:
        Seed used for reproducible MCAR deletion. Eight independent runs per
        technique and scenario use incrementing seeds from this value.
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
            Averaged results table (one row per technique x missing_pct,
            including successful repetition count), also written to
            ``results_dir / 'results.csv'``.
        """
        self._loader.load()
        ground_truth = self._loader.select_column(self.target_column)

        all_results: list[MetricsResult] = []

        for missing_pct in self.missing_rates:
            try:
                self._generator.generate(ground_truth, missing_pct)
            except InvalidMissingRateError as exc:
                logger.warning(
                    "Skipping %.0f%% missingness because it cannot be generated: %s",
                    missing_pct * 100,
                    exc,
                )
                continue

            for imputer in self.imputers:
                try:
                    metrics_result, outcome, removed_indices = run_repeated_imputation(
                        imputer=imputer,
                        ground_truth=ground_truth,
                        missing_pct=missing_pct,
                        random_state=self.random_state,
                    )
                except ImputationError as exc:
                    logger.error(
                        "Skipping %s at %.0f%% missing due to error: %s",
                        imputer.name,
                        missing_pct * 100,
                        exc,
                    )
                    continue

                all_results.append(metrics_result)

                if generate_plots and missing_pct == self.missing_rates[0]:
                    self._visualizer.plot_original_vs_imputed(
                        ground_truth=ground_truth,
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
