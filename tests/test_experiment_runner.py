"""Tests for end-to-end experiment rate handling."""

from __future__ import annotations

import pandas as pd

from src.experiment_runner import ExperimentRunner
from src.imputer_base import BaseImputer
from src.makima_imputer import MakimaImputer
from src.utils import DEFAULT_MISSING_RATES


class RecordingImputer(BaseImputer):
    name = "Recording"
    missing_masks: list[tuple[int, ...]] = []

    def _impute(self, series: pd.Series) -> pd.Series:
        self.missing_masks.append(tuple(series.index[series.isna()]))
        return series.fillna(series.mean())


def test_runner_skips_99_percent_when_dataset_is_too_small(tmp_path):
    data_path = tmp_path / "small.csv"
    pd.DataFrame({"value": range(30)}).to_csv(data_path, index=False)
    runner = ExperimentRunner(
        data_path=data_path,
        target_column="value",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        missing_rates=(0.99,),
        imputers=[MakimaImputer()],
    )

    results = runner.run(generate_plots=False)

    assert results.empty


def test_default_runner_covers_paper_stress_increments(tmp_path):
    data_path = tmp_path / "series.csv"
    pd.DataFrame({"value": range(500)}).to_csv(data_path, index=False)
    runner = ExperimentRunner(
        data_path=data_path,
        target_column="value",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        imputers=[MakimaImputer()],
    )

    results = runner.run(generate_plots=False)

    expected_rates = (0.01, 0.03, 0.05, *(pct / 100 for pct in range(10, 100, 10)), 0.99)
    assert DEFAULT_MISSING_RATES == expected_rates
    assert tuple(results["missing_pct"]) == expected_rates
    assert tuple(results["n_runs"]) == (8,) * len(expected_rates)


def test_runner_performs_eight_independent_repetitions(tmp_path):
    data_path = tmp_path / "series.csv"
    pd.DataFrame({"value": range(500)}).to_csv(data_path, index=False)
    RecordingImputer.missing_masks = []
    runner = ExperimentRunner(
        data_path=data_path,
        target_column="value",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        missing_rates=(0.10,),
        imputers=[RecordingImputer()],
    )

    results = runner.run(generate_plots=True)

    assert len(RecordingImputer.missing_masks) == 8
    assert len(set(RecordingImputer.missing_masks)) == 8
    assert results.loc[0, "n_runs"] == 8
