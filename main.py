"""Optional backend entry point for running the imputation experiment
pipeline without the Streamlit UI (e.g. for scripted/batch/CI use).

Usage
-----
    python main.py --data data/example.csv --column voltage_a

If no arguments are given, defaults are used and a synthetic example
dataset is generated automatically under ``data/`` for demonstration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_runner import ExperimentRunner
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_example_dataset(path: Path, n_points: int = 2000, seed: int = 7) -> None:
    """Create a synthetic smart-grid-like CSV dataset for demonstration."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    voltage_a = 220 + 3 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.8, n_points)
    current_a = 15 + 8 * np.sin(2 * np.pi * t / 96 + 0.5) + rng.normal(0, 1.2, n_points)
    temperature = 25 + 5 * np.sin(2 * np.pi * t / (96 * 30)) + rng.normal(0, 0.3, n_points)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-06-09", periods=n_points, freq="min"),
            "voltage_a": voltage_a,
            "current_a": current_a,
            "temperature": temperature,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Generated example dataset at %s (%d rows).", path, len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Smart Grid imputation experiment pipeline.")
    parser.add_argument("--data", type=str, default="data/example_smart_grid.csv", help="Path to CSV dataset.")
    parser.add_argument("--column", type=str, default="current_a", help="Numeric column to impute.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for results.csv.")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Directory for saved plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        logger.info("Dataset %s not found; generating a synthetic example dataset.", data_path)
        _generate_example_dataset(data_path)

    runner = ExperimentRunner(
        data_path=data_path,
        target_column=args.column,
        results_dir=Path(args.results_dir),
        figures_dir=Path(args.figures_dir),
    )
    results_df = runner.run(generate_plots=not args.no_plots)

    print("\n=== Experiment Summary (mean by technique) ===")
    print(
        results_df.groupby("technique")[["mae", "rmse", "relative_error", "mape", "execution_time_seconds"]]
        .mean()
        .round(4)
        .sort_values("relative_error")
    )


if __name__ == "__main__":
    main()
