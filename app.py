"""SmartGrid-Imputation — Streamlit dashboard.

Presentation layer only. All data processing, imputation, and metric
computation is delegated to the ``src`` package; this file wires user
interactions to those backend functions and renders the results.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DataLoader
from src.experiment_runner import ExperimentRunner, IMPUTER_REGISTRY, get_imputer
from src.metrics import MetricsCalculator
from src.missing_generator import MissingGenerator
from src.plotly_visuals import (
    plot_correlation_heatmap,
    plot_error_vs_missing_pct,
    plot_metric_comparison,
    plot_missing_distribution,
    plot_original_vs_imputed,
    plot_runtime_comparison,
)
from src.utils import DEFAULT_MISSING_RATES, SmartGridImputationError, ensure_directory

# --------------------------------------------------------------------------- #
# Page configuration & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SmartGrid-Imputation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = ensure_directory(Path("data"))
RESULTS_DIR = ensure_directory(Path("results"))
FIGURES_DIR = ensure_directory(Path("figures"))

CUSTOM_CSS = """
<style>
:root {
    --sg-navy: #1B2A41;
    --sg-teal: #2E9E5B;
    --sg-amber: #E08E45;
    --sg-bg: #F7F5F0;
}
.stApp { background-color: var(--sg-bg); }
section[data-testid="stSidebar"] {
    background-color: var(--sg-navy);
}
section[data-testid="stSidebar"] * { color: #EDEDED !important; }
h1, h2, h3 { color: #1B2A41; }
div[data-testid="stMetricValue"] { color: #2E9E5B; }
.sg-hero {
    background: linear-gradient(120deg, #1B2A41 0%, #2E4057 100%);
    padding: 2rem 2.25rem;
    border-radius: 14px;
    color: #F7F5F0;
    margin-bottom: 1.25rem;
}
.sg-hero h1 { color: #F7F5F0; margin-bottom: 0.25rem; }
.sg-hero p { color: #D7DBE0; font-size: 1.05rem; }
.sg-card {
    background: white;
    border: 1px solid #E4E0D6;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    height: 100%;
}
.sg-badge {
    display: inline-block;
    background: #2E9E5B22;
    color: #206B3D;
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.35rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
_DEFAULTS = {
    "dataframe": None,
    "dataset_name": None,
    "target_column": None,
    "missing_pct": 0.05,
    "series_with_missing": None,
    "ground_truth": None,
    "removed_indices": None,
    "imputation_results": {},  # technique -> ImputationResult
    "results_df": pd.DataFrame(),
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------------------------------- #
# Cached backend calls
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Persist uploaded bytes to disk and load them via DataLoader."""
    temp_path = DATA_DIR / filename
    temp_path.write_bytes(file_bytes)
    loader = DataLoader(temp_path)
    return loader.load()


@st.cache_data(show_spinner=False)
def _generate_example_dataset() -> pd.DataFrame:
    path = DATA_DIR / "example_smart_grid.csv"
    DataLoader.create_example_dataset(path)
    return DataLoader(path).load()


@st.cache_data(show_spinner=False)
def _generate_missing(_series_values: tuple, index_len: int, missing_pct: float, seed: int = 42):
    """Cache-friendly wrapper around MissingGenerator.generate.

    Streamlit's cache requires hashable arguments, so the series is passed
    in as a tuple of values; it is reconstructed into a pandas Series here.
    """
    series = pd.Series(_series_values)
    generator = MissingGenerator(random_state=seed)
    return generator.generate(series, missing_pct)


def zip_directory(directory: Path) -> bytes:
    """Zip all files in a directory (non-recursive) and return the bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(directory.glob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    buffer.seek(0)
    return buffer.read()


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## ⚡ SmartGrid-Imputation")
st.sidebar.caption("Reproducing Schreiber et al., IEEE Access 2023")

PAGES = [
    "🏠 Home",
    "📂 Dataset",
    "⚡ Missing Data Generator",
    "🧠 Imputation Algorithms",
    "📊 Experiment Results",
    "📈 Visualizations",
    "📑 Literature Summary",
    "ℹ️ About",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
if st.session_state.dataframe is not None:
    st.sidebar.success(f"Dataset loaded: **{st.session_state.dataset_name}**")
    st.sidebar.caption(f"{st.session_state.dataframe.shape[0]:,} rows × {st.session_state.dataframe.shape[1]} cols")
else:
    st.sidebar.info("No dataset loaded yet.")

if st.session_state.target_column:
    st.sidebar.caption(f"Target column: **{st.session_state.target_column}**")


# --------------------------------------------------------------------------- #
# 🏠 HOME
# --------------------------------------------------------------------------- #
def render_home() -> None:
    st.markdown(
        """
        <div class="sg-hero">
            <h1>⚡ SmartGrid-Imputation</h1>
            <p>An interactive research dashboard for experimenting with data imputation
            for missing values in Smart Grid electrical time-series data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Abstract")
        st.write(
            "Smart Grids generate massive volumes of sensor data — voltages, currents, "
            "and temperatures — that are frequently corrupted by sensor faults, "
            "transmission errors, and communication failures. This dashboard reproduces "
            "the implementation of **Schreiber et al. (IEEE Access, 2023)**, which "
            "compares four imputation techniques on real underground-substation data "
            "from the CEEE utility (Porto Alegre, Brazil), and validates **Makima** "
            "interpolation as the most robust technique in the paper's 99% stress test. "
            "This dashboard lets you test uploaded datasets from 1% to 99% missingness."
        )

        st.markdown("### Research Objectives")
        st.markdown(
            """
            - Reproduce Median, LOCF, KNN, and Makima imputation as production-quality code.
            - Benchmark each technique across standard MCAR rates and a 99% stress test.
            - Quantify accuracy (MAE, RMSE, Relative Error, MAPE) and runtime.
            - Provide an interactive tool to upload any CSV time series and repeat the study.
            """
        )

        st.markdown("### Algorithms Implemented")
        badge_cols = st.columns(4)
        badges = [
            ("Median", "Windowed central tendency"),
            ("LOCF", "Last Observation Carried Forward"),
            ("KNN", "K-Nearest Neighbor"),
            ("Makima", "Modified Akima interpolation"),
        ]
        for c, (name, desc) in zip(badge_cols, badges):
            with c:
                st.markdown(f"<div class='sg-card'><b>{name}</b><br><small>{desc}</small></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### Workflow")
        steps = [
            "📂 Load & inspect a CSV time series",
            "⚡ Generate MCAR missing values (1–99%)",
            "🧠 Run one or more imputation algorithms",
            "📊 Compare MAE / RMSE / Relative Error / MAPE",
            "📈 Explore interactive visualizations",
            "⬇️ Export results, figures, and imputed data",
        ]
        for i, step in enumerate(steps, start=1):
            st.markdown(f"**{i}.** {step}")

        st.markdown("### Technology Stack")
        st.markdown(
            "`Python 3.12` · `pandas` · `numpy` · `scipy` · `scikit-learn` · "
            "`matplotlib` · `plotly` · `streamlit`"
        )

    st.divider()
    st.markdown("### Quick Instructions")
    st.info(
        "Start on the **📂 Dataset** page to upload a CSV (or load the built-in example), "
        "then move through **⚡ Missing Data Generator → 🧠 Imputation Algorithms → "
        "📊 Experiment Results → 📈 Visualizations** in order. Everything runs directly "
        "in this browser tab — no terminal required."
    )


# --------------------------------------------------------------------------- #
# 📂 DATASET
# --------------------------------------------------------------------------- #
def render_dataset() -> None:
    st.title("📂 Dataset")

    tab_upload, tab_example = st.tabs(["Upload CSV", "Use Example Dataset"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a CSV time-series dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                df = _load_csv(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.dataframe = df
                st.session_state.dataset_name = uploaded_file.name
                st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns.")
            except SmartGridImputationError as exc:
                st.error(f"Failed to load dataset: {exc}")

    with tab_example:
        st.write(
            "Load a synthetic 2,000-point smart-grid dataset with secondary voltages "
            "(A/B/C), primary currents (A/B/C), room temperature, and transformer "
            "temperature — structured like the CEEE id17 substation data used in the paper."
        )
        if st.button("⚡ Load Example Dataset", type="primary"):
            df = _generate_example_dataset()
            st.session_state.dataframe = df
            st.session_state.dataset_name = "example_smart_grid.csv"
            st.success(f"Loaded example dataset — {df.shape[0]:,} rows × {df.shape[1]} columns.")

    df = st.session_state.dataframe
    if df is None:
        st.warning("No dataset loaded yet. Upload a CSV or load the example dataset above.")
        return

    st.divider()
    st.markdown("### Preview")
    st.dataframe(df.head(50), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Total missing cells", int(df.isna().sum().sum()))
    m4.metric("Numeric columns", len(df.select_dtypes(include="number").columns))

    st.markdown("### Column Summary")
    temp_loader = DataLoader.__new__(DataLoader)
    temp_loader.dataframe = df
    temp_loader.filepath = Path(st.session_state.dataset_name or "uploaded.csv")
    st.dataframe(temp_loader.summary(), use_container_width=True)

    st.markdown("### Select Target Column")
    numeric_cols = temp_loader.list_numeric_columns()
    if not numeric_cols:
        st.error("This dataset has no numeric columns to impute.")
        return

    default_idx = numeric_cols.index(st.session_state.target_column) if st.session_state.target_column in numeric_cols else 0
    target = st.selectbox("Numeric column to use for imputation experiments", numeric_cols, index=default_idx)
    st.session_state.target_column = target

    with st.expander("📊 Basic statistics for selected column"):
        st.dataframe(df[[target]].describe().T, use_container_width=True)

    if len(numeric_cols) > 1:
        with st.expander("🔥 Correlation heatmap (numeric columns)"):
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)


# --------------------------------------------------------------------------- #
# ⚡ MISSING DATA GENERATOR
# --------------------------------------------------------------------------- #
def render_missing_data() -> None:
    st.title("⚡ Missing Data Generator")

    if st.session_state.dataframe is None or st.session_state.target_column is None:
        st.warning("Load a dataset and select a target column on the **📂 Dataset** page first.")
        return

    st.write("Mechanism: **MCAR** (Missing Completely At Random) — matches the methodology used in the paper.")

    missing_pct_display = st.slider(
        "Missing data percentage",
        min_value=1,
        max_value=99,
        value=min(99, max(1, int(st.session_state.missing_pct * 100))),
        step=1,
        format="%d%%",
    )
    missing_pct = missing_pct_display / 100

    if st.button("🎲 Generate Missing Values", type="primary"):
        loader = DataLoader.__new__(DataLoader)
        loader.dataframe = st.session_state.dataframe
        loader.filepath = Path(st.session_state.dataset_name or "uploaded.csv")

        try:
            ground_truth = loader.select_column(st.session_state.target_column)
            series_with_missing, truth, removed_indices = _generate_missing(
                tuple(ground_truth.tolist()), len(ground_truth), missing_pct
            )
            series_with_missing.index = ground_truth.index
            truth.index = ground_truth.index

            st.session_state.ground_truth = truth
            st.session_state.series_with_missing = series_with_missing
            st.session_state.removed_indices = removed_indices
            st.session_state.missing_pct = missing_pct
            st.session_state.imputation_results = {}  # invalidate stale results
            st.success(f"Generated {missing_pct_display}% MCAR missingness ({len(removed_indices)} values removed).")
        except SmartGridImputationError as exc:
            st.error(f"Could not generate missing values: {exc}")

    if st.session_state.series_with_missing is None:
        st.info("Click **Generate Missing Values** to create a modified dataset.")
        return

    st.divider()
    series_missing = st.session_state.series_with_missing
    ground_truth = st.session_state.ground_truth
    removed = st.session_state.removed_indices

    m1, m2, m3 = st.columns(3)
    m1.metric("Missing values", len(removed))
    m2.metric("Missing %", f"{st.session_state.missing_pct * 100:.1f}%")
    m3.metric("Total records", len(series_missing))

    col_orig, col_mod = st.columns(2)
    with col_orig:
        st.markdown("**Original dataset (target column)**")
        st.dataframe(ground_truth.rename("value").to_frame(), height=280, use_container_width=True)
    with col_mod:
        st.markdown("**Modified dataset (with NaNs)**")
        st.dataframe(series_missing.rename("value").to_frame(), height=280, use_container_width=True)

    st.markdown("### Missing Value Distribution")
    st.plotly_chart(plot_missing_distribution(series_missing), use_container_width=True)


# --------------------------------------------------------------------------- #
# 🧠 IMPUTATION ALGORITHMS
# --------------------------------------------------------------------------- #
def render_imputation() -> None:
    st.title("🧠 Imputation Algorithms")

    if st.session_state.series_with_missing is None:
        st.warning("Generate missing values on the **⚡ Missing Data Generator** page first.")
        return

    st.write("Select one or more algorithms to run against the current missing-data scenario.")

    algorithm_names = list(IMPUTER_REGISTRY.keys())
    n_cols = min(4, len(algorithm_names)) or 1
    cols = st.columns(n_cols)
    selected: list[str] = []
    for i, name in enumerate(algorithm_names):
        with cols[i % n_cols]:
            if st.checkbox(name, value=True, key=f"algo_{name}"):
                selected.append(name)

    run_clicked = st.button("▶️ Run Selected Algorithms", type="primary", disabled=not selected)

    if run_clicked:
        series_missing = st.session_state.series_with_missing
        ground_truth = st.session_state.ground_truth
        removed_indices = st.session_state.removed_indices
        missing_pct = st.session_state.missing_pct

        progress = st.progress(0.0, text="Starting...")
        results: dict = {}
        metrics_rows = []

        for i, name in enumerate(selected, start=1):
            progress.progress((i - 1) / len(selected), text=f"Running {name}...")
            try:
                imputer = get_imputer(name)
                outcome = imputer.impute(series_missing)
                metrics_result = MetricsCalculator.evaluate(
                    technique=name,
                    missing_pct=missing_pct,
                    imputed_series=outcome.imputed_series,
                    ground_truth=ground_truth,
                    removed_indices=removed_indices,
                    execution_time_seconds=outcome.execution_time_seconds,
                )
                results[name] = outcome
                metrics_rows.append(metrics_result)
                st.toast(f"{name} completed in {outcome.execution_time_seconds:.4f}s", icon="✅")
            except SmartGridImputationError as exc:
                st.error(f"{name} failed: {exc}")
            progress.progress(i / len(selected), text=f"{name} done")

        progress.empty()
        st.session_state.imputation_results = results
        st.session_state.results_df = MetricsCalculator.results_to_dataframe(metrics_rows)
        st.success(f"Completed {len(results)}/{len(selected)} algorithm(s).")

    if st.session_state.imputation_results:
        st.divider()
        st.markdown("### Quick Metrics Preview")
        st.dataframe(
            st.session_state.results_df.style.highlight_min(
                subset=["mae", "rmse", "relative_error", "mape"], color="#2E9E5B44"
            ),
            use_container_width=True,
        )
        st.caption("Full comparison, filtering, and export tools are on the **📊 Experiment Results** page.")


# --------------------------------------------------------------------------- #
# 📊 EXPERIMENT RESULTS
# --------------------------------------------------------------------------- #
def render_results() -> None:
    st.title("📊 Experiment Results")

    st.markdown("### One-Click Full Experiment")
    st.write(
        "Runs **all seven algorithms** across the standard missing-data scenarios "
        "(1%, 3%, 5%, 10%, 20%) plus the 99% stress test on the selected target "
        "column, and saves "
        "`results/results.csv` plus all standard figures to `figures/`."
    )
    if st.session_state.target_column is None:
        st.warning("Select a target column on the **📂 Dataset** page first.")
    elif st.button("🚀 Run All Experiments", type="primary"):
        loader = DataLoader.__new__(DataLoader)
        loader.dataframe = st.session_state.dataframe
        loader.filepath = Path(st.session_state.dataset_name or "uploaded.csv")
        temp_csv = DATA_DIR / "_current_dataset.csv"
        st.session_state.dataframe.to_csv(temp_csv, index=False)

        with st.spinner("Running full experiment across all scenarios..."):
            runner = ExperimentRunner(
                data_path=temp_csv,
                target_column=st.session_state.target_column,
                results_dir=RESULTS_DIR,
                figures_dir=FIGURES_DIR,
            )
            full_results_df = runner.run(generate_plots=True)
        st.session_state.results_df = full_results_df
        expected_combinations = len(IMPUTER_REGISTRY) * len(DEFAULT_MISSING_RATES)
        st.success(
            f"Full experiment complete — {len(full_results_df)} successful "
            "technique×scenario combinations evaluated."
        )
        if len(full_results_df) < expected_combinations:
            st.warning(
                "Some combinations were skipped because the dataset was too small for "
                "a missingness rate or an algorithm could not complete the imputation."
            )

    st.divider()
    results_df = st.session_state.results_df
    if results_df is None or results_df.empty:
        st.info("No results yet. Run algorithms on the **🧠 Imputation Algorithms** page, or use the button above.")
        return

    st.markdown("### Comparison Table")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        technique_filter = st.multiselect(
            "Filter by technique", sorted(results_df["technique"].unique()), default=sorted(results_df["technique"].unique())
        )
    with col_f2:
        pct_options = sorted(results_df["missing_pct"].unique())
        pct_filter = st.multiselect(
            "Filter by missing %", pct_options, default=pct_options, format_func=lambda p: f"{p * 100:.0f}%"
        )

    filtered = results_df[
        results_df["technique"].isin(technique_filter) & results_df["missing_pct"].isin(pct_filter)
    ].sort_values(["missing_pct", "relative_error"])

    sort_col = st.selectbox("Sort by", ["relative_error", "mae", "rmse", "mape", "execution_time_seconds"], index=0)
    filtered = filtered.sort_values(sort_col)

    st.dataframe(
        filtered.style.highlight_min(
            subset=["mae", "rmse", "relative_error", "mape", "execution_time_seconds"], color="#2E9E5B44"
        ),
        use_container_width=True,
    )

    if not filtered.empty:
        best_row = filtered.sort_values("relative_error").iloc[0]
        st.success(
            f"🏆 Best performing technique in current filter: **{best_row['technique']}** "
            f"at {best_row['missing_pct'] * 100:.0f}% missing (Relative Error = {best_row['relative_error']:.2f}%)."
        )

    st.download_button(
        "⬇️ Download filtered results as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="smartgrid_imputation_results.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# 📈 VISUALIZATIONS
# --------------------------------------------------------------------------- #
def render_visualizations() -> None:
    st.title("📈 Visualizations")

    tab_signal, tab_error, tab_runtime, tab_metrics = st.tabs(
        ["Original vs Imputed", "Error vs Missing %", "Runtime Comparison", "Metric Comparison"]
    )

    with tab_signal:
        if st.session_state.series_with_missing is None:
            st.info("Generate missing values first on the **⚡ Missing Data Generator** page.")
        else:
            technique = st.selectbox("Technique", list(IMPUTER_REGISTRY.keys()), key="viz_technique")
            if st.button("Generate plot", key="viz_signal_btn"):
                try:
                    imputer = get_imputer(technique)
                    outcome = imputer.impute(st.session_state.series_with_missing)
                    fig = plot_original_vs_imputed(
                        st.session_state.ground_truth,
                        outcome.imputed_series,
                        st.session_state.removed_indices,
                        technique,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except SmartGridImputationError as exc:
                    st.error(f"{technique} failed: {exc}")

    results_df = st.session_state.results_df

    with tab_error:
        if results_df is None or results_df.empty:
            st.info("Run the full experiment on the **📊 Experiment Results** page first.")
        else:
            metric_choice = st.selectbox("Metric", ["relative_error", "mae", "rmse", "mape"], key="viz_error_metric")
            st.plotly_chart(plot_error_vs_missing_pct(results_df, metric_choice), use_container_width=True)

    with tab_runtime:
        if results_df is None or results_df.empty:
            st.info("Run the full experiment on the **📊 Experiment Results** page first.")
        else:
            st.plotly_chart(plot_runtime_comparison(results_df), use_container_width=True)

    with tab_metrics:
        if results_df is None or results_df.empty:
            st.info("Run the full experiment on the **📊 Experiment Results** page first.")
        else:
            pct_options = ["All scenarios"] + [f"{p * 100:.0f}%" for p in sorted(results_df["missing_pct"].unique())]
            pct_choice = st.selectbox("Missing % scope", pct_options, key="viz_metric_pct")
            pct_value = None if pct_choice == "All scenarios" else float(pct_choice.strip("%")) / 100
            st.plotly_chart(plot_metric_comparison(results_df, pct_value), use_container_width=True)

    st.divider()
    st.markdown("### Export Figures")
    if FIGURES_DIR.exists() and any(FIGURES_DIR.glob("*.png")):
        zip_bytes = zip_directory(FIGURES_DIR)
        st.download_button(
            "⬇️ Download all saved figures (ZIP)",
            data=zip_bytes,
            file_name="smartgrid_imputation_figures.zip",
            mime="application/zip",
        )
    else:
        st.caption("No saved figures yet — run the full experiment on the Results page to generate PNGs.")


# --------------------------------------------------------------------------- #
# 📑 LITERATURE SUMMARY
# --------------------------------------------------------------------------- #
def render_literature() -> None:
    st.title("📑 Literature Summary")

    st.markdown(
        """
        #### Data Imputation Techniques Applied to the Smart Grids Environment
        **Schreiber, A. Sausen, De Campos, P. S. Sausen, Ferreira Filho** ·
        *IEEE Access, Vol. 11, 2023* · DOI: 10.1109/ACCESS.2023.3262188
        """
    )

    with st.expander("Abstract", expanded=True):
        st.write(
            "Smart Grids' monitoring and communication technologies drive rapid growth "
            "in sensor data volume, increasing the incidence of errors and missing "
            "values. This paper models, implements, validates, and compares four data "
            "imputation techniques — KNN, Median, LOCF, and Makima — on real data from "
            "an underground substation operated by CEEE in southern Brazil. Across five "
            "simulation scenarios, Makima consistently achieved the lowest error and was "
            "the only technique to remain stable up to 99% missing data."
        )

    with st.expander("Dataset & Experimental Setup"):
        st.markdown(
            """
            - **Source:** CEEE underground substations, Porto Alegre, Brazil
            - **Substation used:** id17 (of 42 evaluated) — most samples, no interruptions
            - **Window:** 14,427 uninterrupted minutes (June 9–19, 2019)
            - **Sampling rate:** 1 packet / 60 seconds
            - **Variables:** 11 analog measurements — voltages (A/B/C), primary & secondary
              currents (A/B/C), room temperature, transformer temperature
            """
        )

    with st.expander("Five Simulation Scenarios"):
        scenario_df = pd.DataFrame(
            {
                "Scenario": ["S1", "S2", "S3", "S4", "S5"],
                "Description": [
                    "1% PMR — baseline accuracy",
                    "3% PMR — consistency check",
                    "5% PMR — standard real-world fault rate",
                    "10% → 99% PMR — extreme stress test",
                    "Primary current (Phase B) — high-variation signal",
                ],
            }
        )
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    with st.expander("Key Results (ERmed)"):
        results_table = pd.DataFrame(
            {
                "Technique": ["KNN", "Median", "LOCF", "Makima"],
                "1% PMR": ["10.61%", "1.56%", "1.64%", "1.38%"],
                "5% PMR": ["19.14%", "1.61%", "1.68%", "1.43%"],
                "Max tolerated PMR": ["~30%", "~20%", "~10%", "99%"],
                "Error at max PMR": ["108% @ 40%", "FAIL @ 30%", "FAIL @ 20%", "5.22% @ 99%"],
            }
        )
        st.dataframe(results_table, use_container_width=True, hide_index=True)
        st.caption(
            "Makima's mathematical structure — using surrounding known points to build a "
            "smooth cubic interpolant — lets it bridge very large gaps while KNN, Median, "
            "and LOCF all fail or degrade sharply beyond 10–40% missing data."
        )

    with st.expander("Limitations Noted in the Paper"):
        st.markdown(
            """
            - Single substation (id17) — generalisability across other substations untested
            - Short 10-day observation window — seasonal effects not captured
            - Only MCAR missingness simulated — real-world gaps may be MAR/NMAR
            - No deep-learning benchmarks (LSTM, GAN) included
            - Only Relative Mean Error reported in the original study
            - Original implementation is MATLAB-only
            """
        )

    with st.expander("Conclusion"):
        st.write(
            "Makima is validated as the superior technique for the Smart Grids "
            "environment: lowest error across all five scenarios, and the only method "
            "that completed the extreme 99%-missing stress test without failure."
        )


# --------------------------------------------------------------------------- #
# ℹ️ ABOUT
# --------------------------------------------------------------------------- #
def render_about() -> None:
    st.title("ℹ️ About")

    st.markdown(
        """
        **SmartGrid-Imputation** is a production-quality Python package and Streamlit
        dashboard that reproduces the implementation section of *Data Imputation
        Techniques Applied to the Smart Grids Environment* (IEEE Access, 2023).

        ### Architecture
        - `src/` — all business logic: data loading, MCAR generation, four imputers
          (each inheriting a common abstract base class), metrics, matplotlib and
          Plotly visualizations, and the experiment orchestrator.
        - `app.py` — presentation layer only; calls backend functions and renders results.
        - `main.py` — optional headless entry point for scripted/batch runs.
        - `tests/` — pytest unit tests for metrics, imputers, and missing-value generation.

        ### Technology Stack
        Python 3.12 · pandas · numpy · scipy · scikit-learn · matplotlib · plotly ·
        streamlit · pytest

        ### Future Work
        1. Deploy Makima in a decentralized mode directly on acquisition hardware.
        2. Integrate with a live monitoring system.
        3. Validate across additional substations and electrical profiles.
        4. Explore hybrid Makima + ML approaches for MAR/NMAR missingness.
        """
    )
    st.caption("Built as a research-internship-style implementation exercise. Not affiliated with CEEE or the paper's authors.")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
_ROUTES = {
    "🏠 Home": render_home,
    "📂 Dataset": render_dataset,
    "⚡ Missing Data Generator": render_missing_data,
    "🧠 Imputation Algorithms": render_imputation,
    "📊 Experiment Results": render_results,
    "📈 Visualizations": render_visualizations,
    "📑 Literature Summary": render_literature,
    "ℹ️ About": render_about,
}

_ROUTES[page]()
