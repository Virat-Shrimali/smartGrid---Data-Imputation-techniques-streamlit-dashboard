"""Interactive Plotly figure builders used by the Streamlit dashboard.

Kept separate from ``app.py`` so that all figure-construction logic (i.e.
business/presentation logic) lives inside ``src/``, and the Streamlit
layer only calls these functions and renders their output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# A calm, utility-grid inspired palette: deep teal/navy base with amber
# and coral accents for contrast, avoiding default Plotly blues.
PALETTE = {
    "original": "#2E4057",
    "imputed": "#E08E45",
    "highlight": "#C1432E",
    "Median": "#3D8A8A",
    "LOCF": "#7B6D8D",
    "KNN": "#C1432E",
    "Makima": "#2E9E5B",
    "PCHIP": "#4C6EF5",
    "MICE": "#B5838D",
    "MissForest": "#8C6D31",
    "grid_bg": "#F7F5F0",
}


def plot_original_vs_imputed(
    ground_truth: pd.Series,
    imputed_series: pd.Series,
    removed_indices: np.ndarray,
    technique: str,
) -> go.Figure:
    """Interactive line chart of original vs imputed signal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ground_truth.index,
            y=ground_truth.values,
            mode="lines",
            name="Original",
            line=dict(color=PALETTE["original"], width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=imputed_series.index,
            y=imputed_series.values,
            mode="lines",
            name="Imputed (full series)",
            line=dict(color=PALETTE["imputed"], width=1.2, dash="dash"),
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=imputed_series.index[removed_indices],
            y=imputed_series.iloc[removed_indices],
            mode="markers",
            name="Imputed points",
            marker=dict(color=PALETTE["highlight"], size=6, symbol="circle-open", line=dict(width=2)),
        )
    )
    fig.update_layout(
        title=f"{technique} — Original vs Imputed Signal",
        xaxis_title="Time index",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def plot_missing_distribution(series_with_missing: pd.Series) -> go.Figure:
    """Bar chart showing where missing values occur along the series index."""
    missing_mask = series_with_missing.isna().astype(int)
    fig = px.bar(
        x=missing_mask.index,
        y=missing_mask.values,
        labels={"x": "Time index", "y": "Missing (1) / Present (0)"},
        title="Missing Value Distribution",
        color_discrete_sequence=[PALETTE["highlight"]],
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, b=40, l=40, r=20), bargap=0)
    return fig


def plot_error_vs_missing_pct(results_df: pd.DataFrame, metric: str = "relative_error") -> go.Figure:
    """Line chart of an error metric vs missing-data percentage per technique."""
    df = results_df.copy()
    df["missing_pct_display"] = df["missing_pct"] * 100

    fig = px.line(
        df.sort_values(["technique", "missing_pct"]),
        x="missing_pct_display",
        y=metric,
        color="technique",
        markers=True,
        color_discrete_map=PALETTE,
        labels={"missing_pct_display": "Missing data (%)", metric: metric.replace("_", " ").title()},
        title=f"{metric.replace('_', ' ').title()} vs Missing Percentage",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, b=40, l=40, r=20))
    return fig


def plot_runtime_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Bar chart of average execution time per technique."""
    summary = (
        results_df.groupby("technique")["execution_time_seconds"]
        .mean()
        .reset_index()
        .sort_values("execution_time_seconds")
    )
    fig = px.bar(
        summary,
        x="technique",
        y="execution_time_seconds",
        color="technique",
        color_discrete_map=PALETTE,
        text_auto=".4f",
        labels={"execution_time_seconds": "Time (seconds)", "technique": "Technique"},
        title="Average Execution Time by Technique",
    )
    fig.update_layout(template="plotly_white", showlegend=False, margin=dict(t=60, b=40, l=40, r=20))
    return fig


def plot_metric_comparison(results_df: pd.DataFrame, missing_pct: float | None = None) -> go.Figure:
    """Grouped bar chart comparing MAE, RMSE, Relative Error, and MAPE."""
    df = results_df if missing_pct is None else results_df[results_df["missing_pct"] == missing_pct]
    summary = df.groupby("technique")[["mae", "rmse", "relative_error", "mape"]].mean().reset_index()
    melted = summary.melt(id_vars="technique", var_name="metric", value_name="value")

    title_suffix = f" @ {missing_pct * 100:.0f}% missing" if missing_pct is not None else " (averaged across scenarios)"
    fig = px.bar(
        melted,
        x="technique",
        y="value",
        color="metric",
        barmode="group",
        labels={"value": "Metric value", "technique": "Technique", "metric": "Metric"},
        title=f"Metric Comparison{title_suffix}",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, b=40, l=40, r=20))
    return fig


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> go.Figure:
    """Correlation heatmap across numeric columns of the dataset."""
    numeric_df = dataframe.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap (numeric columns)",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, b=40, l=40, r=20))
    return fig
