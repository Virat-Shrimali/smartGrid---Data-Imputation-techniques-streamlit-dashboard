"""Tests for end-to-end experiment rate handling."""

from __future__ import annotations

import pandas as pd

from src.experiment_runner import ExperimentRunner
from src.makima_imputer import MakimaImputer


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
