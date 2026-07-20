"""CSV, Plotly HTML, and combined ZIP exports."""

from __future__ import annotations

import io
import zipfile
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go

from models.data_models import RegressionResult


SUMMARY_COLUMNS = [
    "file_name",
    "file_hash",
    "analysis_mode",
    "concentration",
    "concentration_unit",
    "concentration_base_value",
    "scan_rate_V_s",
    "cycle_selection",
    "cycle_number",
    "sweep_direction",
    "peak_type",
    "peak_potential_V",
    "raw_peak_current_A",
    "corrected_peak_current_A",
    "raw_response",
    "blank_response",
    "blank_corrected_response",
    "display_peak_current",
    "display_current_unit",
    "fixed_potential_V",
    "integrated_charge_C",
    "included_in_regression",
    "warning",
]
PROCESSED_COLUMNS = [
    "file_name",
    "concentration",
    "concentration_unit",
    "scan_rate_V_s",
    "cycle",
    "segment",
    "sweep_direction",
    "point_number",
    "elapsed_time_s",
    "potential_V",
    "raw_current_A",
    "smoothed_current_A",
    "baseline_A",
    "corrected_current_A",
]
REGRESSION_COLUMNS = [
    "analysis_mode",
    "x_variable",
    "x_unit",
    "y_variable",
    "y_unit",
    "slope",
    "intercept",
    "r_squared",
    "adjusted_r_squared",
    "slope_standard_error",
    "intercept_standard_error",
    "p_value",
    "rmse",
    "n",
    "equation",
]


def normalized_frame(data: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    """Return a stable export schema even when analysis produced no rows."""

    frame = pd.DataFrame() if data is None else data.copy()
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    return frame[columns]


def summary_csv(data: pd.DataFrame | None) -> bytes:
    return normalized_frame(data, SUMMARY_COLUMNS).to_csv(index=False).encode("utf-8-sig")


def processed_csv(data: pd.DataFrame | None) -> bytes:
    return normalized_frame(data, PROCESSED_COLUMNS).to_csv(index=False).encode("utf-8-sig")


def regression_frame(result: RegressionResult | None) -> pd.DataFrame:
    if result is None:
        return normalized_frame(None, REGRESSION_COLUMNS)
    values = asdict(result)
    return normalized_frame(pd.DataFrame([values]), REGRESSION_COLUMNS)


def regression_csv(result: RegressionResult | None) -> bytes:
    return regression_frame(result).to_csv(index=False).encode("utf-8-sig")


def figure_html(figure: go.Figure | None) -> bytes:
    if figure is None:
        return b"<!doctype html><html><body><p>No plot was generated.</p></body></html>"
    return figure.to_html(include_plotlyjs=True, full_html=True).encode("utf-8")


def analysis_zip(
    summary: pd.DataFrame | None,
    processed: pd.DataFrame | None,
    regression: RegressionResult | None,
    cv_figure: go.Figure | None,
    analysis_figure: go.Figure | None,
) -> bytes:
    """Create the complete downloadable analysis package in memory."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.csv", summary_csv(summary))
        archive.writestr("processed_cv_data.csv", processed_csv(processed))
        archive.writestr("regression_results.csv", regression_csv(regression))
        archive.writestr("cv_overlay.html", figure_html(cv_figure))
        archive.writestr("calibration_plot.html", figure_html(analysis_figure))
    return buffer.getvalue()
