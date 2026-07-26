"""Static (matplotlib) visualization utilities.

Generates and saves the four required plot types:

1. Original vs Imputed signal
2. Error vs Missing Percentage
3. Runtime comparison
4. Metric comparison bar chart

Used by :class:`src.experiment_runner.ExperimentRunner` for headless
figure generation, and reused (with Plotly equivalents) by the Streamlit
app for interactive display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend; safe for scripts and Streamlit's server process.
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_directory, get_logger

logger = get_logger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titleweight": "bold",
    }
)


@dataclass
class Visualizer:
    """Creates and persists standard comparison plots to disk.

    Attributes
    ----------
    output_dir:
        Directory where PNG figures will be saved.
    """

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = ensure_directory(Path(self.output_dir))

    def plot_original_vs_imputed(
        self,
        ground_truth: pd.Series,
        imputed_series: pd.Series,
        removed_indices,
        technique: str,
        missing_pct: float,
        filename: str | None = None,
    ) -> Path:
        """Plot the original signal against the imputed signal.

        Points that were imputed are highlighted with markers.
        """
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(ground_truth.index, ground_truth.values, label="Original", color="#1f77b4", linewidth=1.2)
        ax.plot(
            imputed_series.index,
            imputed_series.values,
            label="Imputed",
            color="#d62728",
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
        ax.scatter(
            imputed_series.index[removed_indices],
            imputed_series.iloc[removed_indices],
            color="#d62728",
            s=14,
            zorder=5,
            label="Imputed points",
        )
        ax.set_title(f"{technique} — Original vs Imputed ({missing_pct * 100:.0f}% missing)")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        fig.tight_layout()

        filename = filename or f"original_vs_imputed_{technique.lower()}_{int(missing_pct * 100)}pct.png"
        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
        return path

    def plot_error_vs_missing_pct(self, results_df: pd.DataFrame, metric: str = "relative_error", filename: str | None = None) -> Path:
        """Plot a chosen error metric against missing-data percentage per technique."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for technique, group in results_df.groupby("technique"):
            group_sorted = group.sort_values("missing_pct")
            ax.plot(
                group_sorted["missing_pct"] * 100,
                group_sorted[metric],
                marker="o",
                label=technique,
            )
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Missing Percentage")
        ax.set_xlabel("Missing data (%)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(loc="best")
        fig.tight_layout()

        filename = filename or f"error_vs_missing_pct_{metric}.png"
        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
        return path

    def plot_runtime_comparison(self, results_df: pd.DataFrame, filename: str = "runtime_comparison.png") -> Path:
        """Bar chart comparing mean execution time per technique."""
        summary = results_df.groupby("technique")["execution_time_seconds"].mean().sort_values()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(summary.index, summary.values, color="#2ca02c")
        ax.set_title("Average Execution Time by Technique")
        ax.set_xlabel("Technique")
        ax.set_ylabel("Time (seconds)")
        ax.bar_label(bars, fmt="%.4f")
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
        return path

    def plot_metric_comparison(self, results_df: pd.DataFrame, missing_pct: float | None = None, filename: str = "metric_comparison.png") -> Path:
        """Grouped bar chart comparing MAE, RMSE, Relative Error, and MAPE."""
        data = results_df
        if missing_pct is not None:
            data = data[data["missing_pct"] == missing_pct]

        summary = data.groupby("technique")[["mae", "rmse", "relative_error", "mape"]].mean()

        fig, ax = plt.subplots(figsize=(9, 5))
        summary.plot(kind="bar", ax=ax)
        title_suffix = f" @ {missing_pct * 100:.0f}% missing" if missing_pct is not None else " (all scenarios)"
        ax.set_title(f"Metric Comparison{title_suffix}")
        ax.set_xlabel("Technique")
        ax.set_ylabel("Metric value")
        ax.legend(title="Metric")
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
        return path
