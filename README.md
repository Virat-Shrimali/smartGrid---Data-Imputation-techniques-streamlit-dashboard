# ⚡ SmartGrid-Imputation

A production-quality Python package and Streamlit dashboard reproducing the
implementation section of:

> Schreiber, J. F., Sausen, A., De Campos, M., Sausen, P. S., & Ferreira Filho, M. T. S.
> (2023). *Data Imputation Techniques Applied to the Smart Grids Environment.*
> **IEEE Access, 11**, 31931–31940. https://doi.org/10.1109/ACCESS.2023.3262188

The paper compares four imputation techniques for missing values in Smart Grid
electrical time-series data — **Median**, **LOCF**, **KNN**, and **Makima** — and
validates Makima as the most robust technique, tolerating up to 99% missing data.

This project reproduces those four techniques as a modular, tested Python package
(`src/`) and exposes them through a polished, multi-page Streamlit dashboard
(`app.py`) — no command-line interaction required.

The package has since been extended with three additional techniques —
**PCHIP**, **MICE**, and **MissForest** — spanning three distinct imputation
families (advanced interpolation, statistical iterative imputation, and
machine-learning imputation). See [Additional Algorithms](#additional-algorithms)
below for details.

---

## Project Description

Smart Grids generate enormous volumes of sensor data (voltage, current,
temperature) that are frequently corrupted by sensor faults, network errors,
and transmission glitches. This package:

- Loads any CSV time-series dataset and lets you select a numeric column.
- Generates MCAR (Missing Completely At Random) missing values at 1%, 3%, 5%,
  10%, and 20% rates.
- Runs seven imputation algorithms — Median, LOCF, KNN, Makima, PCHIP, MICE,
  and MissForest — each inheriting a common abstract base class.
- Computes MAE, RMSE, Relative Error, MAPE, and execution time for every run.
- Produces static (matplotlib) and interactive (Plotly) visualizations.
- Provides a complete Streamlit dashboard for end-to-end, no-code experimentation.

---

## Installation

Requires **Python 3.12+**.

```bash
git clone <this-repository>
cd SmartGrid-Imputation
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Option 1 — Streamlit dashboard (recommended)

```bash
streamlit run app.py
```

Then, in the browser:

1. **📂 Dataset** — upload a CSV or load the built-in example dataset; select a
   numeric target column.
2. **⚡ Missing Data Generator** — choose a missing-data percentage and generate
   MCAR gaps.
3. **🧠 Imputation Algorithms** — select and run one or more algorithms.
4. **📊 Experiment Results** — view the comparison table, or click
   **🚀 Run All Experiments** to benchmark all four techniques across every
   scenario (1–20% missing) in one click.
5. **📈 Visualizations** — explore original-vs-imputed signals, error curves,
   runtime, and metric comparisons; export figures as a ZIP.
6. **📑 Literature Summary** / **ℹ️ About** — background on the source paper and
   this implementation.

### Option 2 — Headless / scripted run

```bash
python main.py --data data/example_smart_grid.csv --column primary_current_a
```

This runs the full experiment pipeline (all four techniques × five missing
rates), writes `results/results.csv`, and saves all figures to `figures/`.

### Running tests

```bash
pytest tests/ -v
```

---

## Folder Structure

```
SmartGrid-Imputation/
│
├── README.md
├── requirements.txt
├── main.py                  # Headless backend entry point
├── app.py                   # Streamlit dashboard (presentation layer only)
│
├── data/                    # Input CSV datasets
├── results/                 # results.csv output
├── figures/                 # Saved PNG plots
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CSV loading & validation
│   ├── missing_generator.py # MCAR missing-value generation
│   ├── imputer_base.py      # Abstract base class + timing/validation
│   ├── median_imputer.py    # Windowed median imputation
│   ├── locf_imputer.py      # Last Observation Carried Forward
│   ├── knn_imputer.py       # KNN imputation (lag-window features)
│   ├── makima_imputer.py    # Modified Akima (Makima) interpolation
│   ├── pchip_imputer.py     # Piecewise Cubic Hermite (PCHIP) interpolation
│   ├── mice_imputer.py      # MICE (IterativeImputer + BayesianRidge)
│   ├── missforest_imputer.py # MissForest (iterative Random Forest)
│   ├── metrics.py           # MAE, RMSE, Relative Error, MAPE
│   ├── visualizer.py        # Static matplotlib figures
│   ├── plotly_visuals.py    # Interactive Plotly figures (for the dashboard)
│   ├── experiment_runner.py # End-to-end pipeline orchestration
│   └── utils.py             # Logging, custom exceptions, shared constants
│
└── tests/
    ├── test_metrics.py
    ├── test_imputers.py
    └── test_missing_generator.py
```

---

## Example Outputs

Running the full experiment produces a `results/results.csv` table such as:

| technique | missing_pct | mae   | rmse  | relative_error | mape  | execution_time_seconds |
|-----------|-------------|-------|-------|-----------------|-------|--------------------------|
| Makima    | 0.01        | 1.15  | 1.53  | 9.69            | 9.69  | 0.0011                   |
| Median    | 0.01        | 1.03  | 1.26  | 8.63            | 8.63  | 0.0033                   |
| LOCF      | 0.01        | 1.30  | 1.71  | 11.27           | 11.27 | 0.0003                   |
| KNN       | 0.01        | 1.11  | 1.37  | 9.30            | 9.30  | 0.2197                   |

...alongside PNG figures in `figures/`: original-vs-imputed signal plots,
error-vs-missing-percentage curves, runtime comparison bars, and grouped
metric comparison charts.

---

## Additional Algorithms

Three techniques were added after the initial implementation, each representing
a distinct imputation family. All three inherit the same `BaseImputer`
abstract class as Median/LOCF/KNN/Makima and are automatically picked up by
the experiment runner, comparison tables, and every plot in the dashboard.

### PCHIP — Piecewise Cubic Hermite Interpolating Polynomial

**Family:** Advanced interpolation.
**Implementation:** `scipy.interpolate.PchipInterpolator` (`src/pchip_imputer.py`).

PCHIP fits a shape-preserving cubic Hermite spline through the known data
points. Unlike a standard cubic spline (or even Makima in some cases), PCHIP
is guaranteed to be **monotonicity-preserving**: it never overshoots between
two points that are both increasing or both decreasing.

- **Advantages:** No overshoot; preserves monotonic trends exactly; smooth
  (C¹) and computationally cheap (`O(n)` per interpolation).
- **Disadvantages:** Less smooth than Makima at inflection points (only
  first-derivative continuous); can look "kinked" on highly oscillatory
  signals where Makima's higher-order smoothing performs better.
- **Computational complexity:** `O(n)` to fit, `O(log n)` per query point —
  the fastest of the seven techniques in practice.
- **Assumptions:** Data is a well-ordered 1D sequence (time series); at
  least 2 valid points are required; performs best when local monotonic
  segments are meaningful (e.g. voltage/current ramps), and provides no
  benefit over Makima on noisy, non-monotonic signals.

### MICE — Multiple Imputation by Chained Equations

**Family:** Statistical iterative imputation.
**Implementation:** `sklearn.impute.IterativeImputer` with a `BayesianRidge`
estimator (`src/mice_imputer.py`).

Because `IterativeImputer` needs several correlated columns to model each
missing value as a function of the others, the univariate series is first
expanded into a lag-window feature matrix (each row = a value plus its
`n_lags` neighbours on either side — the same trick used by `KNNImputer`).
MICE then models each column as a linear function (with a Bayesian Ridge
prior) of the remaining columns, cycling through columns until the
estimates stabilize (`max_iter`, default 10) or `tol` (default 1e-3) is
reached.

- **Advantages:** Principled statistical foundation; captures linear
  correlations between neighbouring time points; provides implicit
  uncertainty estimates via the Bayesian Ridge posterior; well-studied and
  widely used in applied statistics.
- **Disadvantages:** Assumes approximately linear relationships between
  features (a Bayesian *Ridge* is a linear model); slower than simple
  interpolation; can be sensitive to poorly conditioned feature windows.
- **Computational complexity:** `O(max_iter × n_lags × n)` roughly, since
  each iteration fits a Bayesian Ridge regression (itself `O(n × p²)` for
  `p` features) per chained-equation pass.
- **Assumptions:** Data is Missing At Random (MAR) with respect to the
  engineered lag features; relationships between lagged values are
  reasonably linear; at least 2 valid points are required.

### MissForest

**Family:** Machine-learning imputation.
**Implementation:** `src/missforest_imputer.py`. If the third-party
`missforest` package (and its own dependencies) is installed and
importable, it is used directly. Otherwise — the default in this
project's `requirements.txt`, to avoid pulling in extra dependencies
(`lightgbm`, `tqdm`, etc.) — a self-contained fallback reproduces the
core MissForest algorithm (Stekhoven & Bühlmann, 2012) using
`sklearn.ensemble.RandomForestRegressor`:

1. Missing cells are initialized with their column median.
2. Columns are processed in ascending order of missingness.
3. For each column, a `RandomForestRegressor` is trained on the other
   (currently-imputed) columns and used to re-predict that column's
   missing cells.
4. Steps 2–3 repeat for up to `max_iter` passes (default 3), stopping
   early once the relative change between iterations falls below `tol`.

Only numeric columns are supported (matching the rest of this package),
so the categorical-variable branch of the original MissForest algorithm
(which would use a `RandomForestClassifier`) is not implemented here.

- **Advantages:** Captures non-linear relationships between neighbouring
  time points; robust to outliers and mixed value scales; in benchmarks
  above, achieved the lowest MAE/RMSE/Relative Error of all seven
  techniques on the smooth synthetic example dataset.
- **Disadvantages:** By far the slowest technique (several seconds to
  tens of seconds per run, vs. milliseconds for interpolation-based
  methods); performance is sensitive to `n_estimators`/`max_depth`;
  provides no smoothness or monotonicity guarantees between imputed
  points.
- **Computational complexity:** `O(max_iter × n_lags × n_estimators × n log n)`
  — each of the `max_iter` passes retrains a forest of `n_estimators`
  trees per column with missing data, and each tree costs `O(n log n)`
  to build.
- **Assumptions:** Data is Missing At Random (MAR); relationships between
  lagged features may be non-linear; at least 2 valid points are
  required; runtime scales with dataset size, so very long series should
  use a reduced `n_estimators`/`max_iter` for interactive use.

### Configurable Parameters

| Technique  | Parameter      | Default | Description                                   |
|------------|----------------|---------|------------------------------------------------|
| PCHIP      | *(none)*       | —       | Behaviour is fully determined by the data.      |
| MICE       | `max_iter`     | 10      | Max chained-equation rounds.                    |
| MICE       | `random_state` | 42      | Reproducibility seed.                           |
| MICE       | `tol`          | 1e-3    | Convergence tolerance.                          |
| MICE       | `n_lags`       | 4       | Lag/lead columns built per side.                |
| MissForest | `n_estimators` | 50      | Trees per Random Forest.                        |
| MissForest | `max_depth`    | None    | Max tree depth (unlimited by default).          |
| MissForest | `max_iter`     | 3       | Max iterative refinement passes.                |
| MissForest | `random_state` | 42      | Reproducibility seed.                           |

## Future Work

- Deploy the best-performing technique (Makima) in a decentralized mode,
  directly on substation acquisition hardware.
- Integrate with a live monitoring system for real-time gap-filling.
- Validate across additional substations and electrical profiles.
- Explore hybrid Makima + machine-learning approaches for MAR/NMAR
  (non-random) missingness patterns, which this MCAR-based study does not cover.
- Add deep-learning benchmarks (LSTM, GAN-based imputers) for comparison.

---

## License / Attribution

This is an independent, educational reproduction of the methodology described
in the referenced IEEE Access paper. It is not affiliated with the paper's
authors, UNIJUÍ, or CEEE-D.
