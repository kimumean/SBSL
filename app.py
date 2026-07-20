"""Streamlit user interface for VersaStudio CV analysis."""

from __future__ import annotations

import hmac
import os
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cv_analysis import (
    CONCENTRATION_TO_M,
    CURRENT_UNIT_FACTORS,
    combine_peak_response,
    concentration_base_value,
    convert_concentration,
    convert_current,
    current_at_potential,
    find_peak,
    integrate_charge,
    interval_mean_current,
    preprocess_cv,
    transform_scan_rate,
)
from core.cycle_detection import annotate_cycles, select_cycles
from core.exporters import (
    analysis_zip,
    figure_html,
    processed_csv,
    regression_csv,
    summary_csv,
)
from core.par_parser import metadata_value, parse_par_bytes
from core.regression import (
    aggregate_replicates,
    linear_regression,
    regression_confidence_band,
)
from models.data_models import CVFileData, FileInput, PeakResult, RegressionResult, preserve_file_input


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 30
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_PLOT_POINTS_PER_TRACE = 10_000
CONCENTRATION_UNITS = ["M", "mM", "µM", "nM", "ppm", "Custom"]


st.set_page_config(
    page_title="CV Calibration & Scan-Rate Analyzer",
    page_icon="⚗️",
    layout="wide",
)


def _configured_password() -> str | None:
    password = os.getenv("APP_PASSWORD")
    if password:
        return password
    try:
        return str(st.secrets["APP_PASSWORD"]) if "APP_PASSWORD" in st.secrets else None
    except Exception:
        return None


def require_authentication() -> None:
    """Show a login only when the operator configured APP_PASSWORD."""

    expected = _configured_password()
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("CV Calibration & Scan-Rate Analyzer")
    st.info("This deployment requires a password.")
    supplied = st.text_input("Password", type="password")
    if st.button("Sign in", type="primary"):
        if hmac.compare_digest(supplied, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "parsed_files": {},
        "file_inputs": {},
        "parse_errors": [],
        "dismissed_ids": set(),
        "uploader_version": 0,
        "analysis_mode": "Concentration series",
        "analysis_outputs": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def default_input(parsed: CVFileData) -> FileInput:
    scan_rate = metadata_value(parsed.metadata, "Scan Rate (V/s)")
    try:
        rate = float(scan_rate) if scan_rate is not None else None
    except (TypeError, ValueError):
        rate = None
    return FileInput(scan_rate_V_s=rate)


def process_uploads(uploaded_files: list[Any] | None) -> None:
    if not uploaded_files:
        return
    parsed_files: dict[str, CVFileData] = st.session_state.parsed_files
    file_inputs: dict[str, dict[str, Any]] = st.session_state.file_inputs
    candidates: list[tuple[Any, bytes]] = []
    for uploaded in uploaded_files:
        contents = uploaded.getvalue()
        if len(contents) > MAX_FILE_BYTES:
            st.session_state.parse_errors.append(f"{uploaded.name}: exceeds the 20 MB per-file limit.")
            continue
        candidates.append((uploaded, contents))
    if len(candidates) > MAX_FILES:
        st.error(f"A session can contain at most {MAX_FILES} files.")
        return
    if sum(len(contents) for _, contents in candidates) > MAX_TOTAL_BYTES:
        st.error("The upload queue exceeds the 200 MB session limit.")
        return
    for uploaded, contents in candidates:
        try:
            parsed = parse_par_bytes(contents, uploaded.name)
            if parsed.file_id in st.session_state.dismissed_ids:
                continue
            if parsed.file_id not in parsed_files and len(parsed_files) >= MAX_FILES:
                st.session_state.parse_errors.append(f"{uploaded.name}: 30-file session limit reached.")
                continue
            if sum(item.file_size for item in parsed_files.values() if item.file_id != parsed.file_id) + parsed.file_size > MAX_TOTAL_BYTES:
                st.session_state.parse_errors.append(f"{uploaded.name}: 200 MB session limit reached.")
                continue
            if parsed.file_id not in parsed_files:
                parsed.raw_data = annotate_cycles(parsed.raw_data, parsed.metadata)
                parsed_files[parsed.file_id] = parsed
                preserve_file_input(file_inputs, parsed.file_id, default_input(parsed))
        except Exception as error:
            message = f"{uploaded.name}: {error}"
            if message not in st.session_state.parse_errors:
                st.session_state.parse_errors.append(message)


def reset_all() -> None:
    st.session_state.parsed_files = {}
    st.session_state.file_inputs = {}
    st.session_state.parse_errors = []
    st.session_state.dismissed_ids = set()
    st.session_state.analysis_outputs = None
    st.session_state.uploader_version += 1


def valid_window(start: float | None, end: float | None) -> tuple[float, float] | None:
    if start is None or end is None or not np.isfinite(start) or not np.isfinite(end) or start == end:
        return None
    return (float(min(start, end)), float(max(start, end)))


def response_requirements(response: str) -> tuple[bool, bool]:
    anodic = response in ("Anodic peak current, Ipa", "abs(Ipa)", "Both", "Ipa - Ipc", "Ipa - abs(Ipc)", "log10(abs(Ipa))")
    cathodic = response in ("Cathodic peak current, Ipc", "abs(Ipc)", "Both", "Ipa - Ipc", "Ipa - abs(Ipc)", "log10(abs(Ipc))")
    return anodic, cathodic


def extract_responses(
    data: pd.DataFrame,
    response: str,
    anodic_window: tuple[float, float] | None,
    cathodic_window: tuple[float, float] | None,
    interval: tuple[float, float] | None,
    fixed_potential: float | None,
    anodic_sweep: str,
    cathodic_sweep: str,
    other_sweep: str,
) -> list[PeakResult]:
    need_anodic, need_cathodic = response_requirements(response)
    anodic: PeakResult | None = None
    cathodic: PeakResult | None = None
    if need_anodic:
        if anodic_window is None:
            raise ValueError("Set an anodic peak-search potential window before running peak analysis.")
        anodic = find_peak(data, "anodic", anodic_window, anodic_sweep)
    if need_cathodic:
        if cathodic_window is None:
            raise ValueError("Set a cathodic peak-search potential window before running peak analysis.")
        cathodic = find_peak(data, "cathodic", cathodic_window, cathodic_sweep)
    if response == "Both":
        return [item for item in (anodic, cathodic) if item is not None]
    if response in ("Anodic peak current, Ipa", "log10(abs(Ipa))"):
        return [anodic] if anodic else []
    if response == "abs(Ipa)" and anodic:
        anodic.response_A = abs(anodic.response_A) if anodic.response_A is not None else None
        anodic.peak_type = "abs(Ipa)"
        return [anodic]
    if response in ("Cathodic peak current, Ipc", "log10(abs(Ipc))"):
        return [cathodic] if cathodic else []
    if response == "abs(Ipc)" and cathodic:
        cathodic.response_A = abs(cathodic.response_A) if cathodic.response_A is not None else None
        cathodic.peak_type = "abs(Ipc)"
        return [cathodic]
    if response in ("Ipa - Ipc", "Ipa - abs(Ipc)") and anodic and cathodic:
        value = combine_peak_response(response, anodic, cathodic)
        return [
            PeakResult(
                peak_type=response,
                peak_potential_V=anodic.peak_potential_V,
                raw_current_A=value,
                corrected_current_A=value,
                response_A=value,
                sweep_direction="combined",
                warnings=anodic.warnings + cathodic.warnings,
            )
        ]
    if response in ("Fixed-potential current", "Baseline-corrected fixed-potential current"):
        if fixed_potential is None:
            raise ValueError("Enter a fixed potential.")
        result = current_at_potential(data, float(fixed_potential), other_sweep)
        if response.startswith("Baseline-corrected"):
            result.peak_type = "baseline-corrected fixed-potential current"
        return [result]
    if response == "Potential interval mean current":
        if interval is None:
            raise ValueError("Enter a valid potential interval.")
        return [interval_mean_current(data, interval, other_sweep)]
    if response == "Integrated charge":
        if interval is None:
            raise ValueError("Enter a valid integration potential interval.")
        return [integrate_charge(data, interval, other_sweep)]
    raise ValueError(f"Unsupported response: {response}")


def create_cv_figure(
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    current_unit: str,
    show_legend: bool,
    show_grid: bool,
    show_zero_line: bool,
    plot_trace: str,
    anodic_window: tuple[float, float] | None,
    cathodic_window: tuple[float, float] | None,
    highlight_file: str | None,
) -> go.Figure:
    figure = go.Figure()
    y_column = {
        "Raw": "raw_current_A",
        "Smoothed": "smoothed_current_A",
        "Baseline-corrected": "corrected_current_A",
    }[plot_trace]
    for (file_name, cycle, sweep), group in processed.groupby(["file_name", "cycle", "sweep_direction"], sort=False):
        stride = max(1, int(np.ceil(len(group) / MAX_PLOT_POINTS_PER_TRACE)))
        shown = group.iloc[::stride]
        width = 4 if file_name == highlight_file else 2
        figure.add_trace(
            go.Scattergl(
                x=shown["potential_V"],
                y=convert_current(shown[y_column], current_unit),
                mode="lines",
                name=f"{file_name} · {cycle} · {sweep}",
                line={"width": width, "dash": "solid" if sweep == "anodic" else "dash"},
                customdata=np.column_stack((
                    shown["file_name"], shown["cycle"], shown["sweep_direction"], shown["point_number"]
                )),
                hovertemplate=(
                    "File: %{customdata[0]}<br>Cycle: %{customdata[1]}<br>"
                    "Sweep: %{customdata[2]}<br>Point: %{customdata[3]}<br>"
                    "E: %{x:.6g} V<br>I: %{y:.6g} " + current_unit + "<extra></extra>"
                ),
            )
        )
    marker_columns = {"peak_potential_V", "corrected_peak_current_A", "file_name", "peak_type"}
    markers = (
        summary.dropna(subset=["peak_potential_V", "corrected_peak_current_A"])
        if marker_columns.issubset(summary.columns)
        else pd.DataFrame()
    )
    if not markers.empty:
        figure.add_trace(
            go.Scatter(
                x=markers["peak_potential_V"],
                y=convert_current(markers["corrected_peak_current_A"], current_unit),
                mode="markers",
                marker={"size": 11, "symbol": "diamond", "color": "#d62728"},
                text=markers["file_name"] + " · " + markers["peak_type"],
                name="Detected response",
                hovertemplate="%{text}<br>E: %{x:.6g} V<br>I: %{y:.6g} " + current_unit + "<extra></extra>",
            )
        )
    if anodic_window:
        figure.add_vrect(x0=anodic_window[0], x1=anodic_window[1], fillcolor="#ff7f0e", opacity=0.08, line_width=0, annotation_text="Anodic window")
    if cathodic_window:
        figure.add_vrect(x0=cathodic_window[0], x1=cathodic_window[1], fillcolor="#1f77b4", opacity=0.08, line_width=0, annotation_text="Cathodic window")
    figure.update_layout(
        title="CV overlay",
        xaxis_title="Potential, E / V",
        yaxis_title=f"Current, I / {current_unit}",
        hovermode="closest",
        showlegend=show_legend,
        template="plotly_white",
        height=620,
    )
    figure.update_xaxes(showgrid=show_grid, zeroline=show_zero_line)
    figure.update_yaxes(showgrid=show_grid, zeroline=show_zero_line)
    return figure


def analysis_rows(settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summary_rows: list[dict[str, Any]] = []
    processed_frames: list[pd.DataFrame] = []
    top_warnings: list[str] = []
    for file_id, parsed in st.session_state.parsed_files.items():
        inputs = st.session_state.file_inputs[file_id]
        if not inputs.get("use", True):
            continue
        traces = select_cycles(parsed.raw_data, inputs.get("cycle_selection", "Last cycle"), settings["last_n"])
        if not traces:
            top_warnings.append(f"{parsed.file_name}: selected cycle is unavailable.")
            continue
        for cycle_label, trace in traces:
            peak_data = preprocess_cv(
                trace,
                smoothing=settings["smoothing"] != "None" and settings["smooth_peak"],
                smoothing_window=settings["smooth_window"],
                smoothing_order=settings["smooth_order"],
                baseline_method=settings["baseline"],
                baseline_order=settings["baseline_order"],
                baseline_regions=settings["baseline_regions"],
                peak_window=settings["anodic_window"] or settings["cathodic_window"],
            )
            plot_data = preprocess_cv(
                trace,
                smoothing=settings["smoothing"] != "None" and settings["smooth_plot"],
                smoothing_window=settings["smooth_window"],
                smoothing_order=settings["smooth_order"],
                baseline_method=settings["baseline"],
                baseline_order=settings["baseline_order"],
                baseline_regions=settings["baseline_regions"],
                peak_window=settings["anodic_window"] or settings["cathodic_window"],
            )
            export = plot_data.copy()
            export["file_name"] = parsed.file_name
            export["concentration"] = inputs.get("concentration")
            export["concentration_unit"] = inputs.get("concentration_unit", "mM")
            export["scan_rate_V_s"] = inputs.get("scan_rate_V_s")
            export["cycle"] = cycle_label
            export["segment"] = export["segment_number"]
            processed_frames.append(export)
            try:
                results = extract_responses(
                    peak_data,
                    settings["response"],
                    settings["anodic_window"],
                    settings["cathodic_window"],
                    settings["interval"],
                    settings["fixed_potential"],
                    settings["anodic_sweep"],
                    settings["cathodic_sweep"],
                    settings["other_sweep"],
                )
            except ValueError as error:
                top_warnings.append(f"{parsed.file_name} · {cycle_label}: {error}")
                continue
            concentration = inputs.get("concentration")
            unit = inputs.get("concentration_unit", "mM")
            base = concentration_base_value(concentration, unit)
            for result in results:
                warnings = list(parsed.warnings) + result.warnings
                if settings["analysis_mode"] == "Concentration series" and base is None:
                    warnings.append("Missing concentration")
                response_value = result.response_A
                if settings["response"].startswith("log10"):
                    if response_value is None or abs(response_value) <= 0:
                        warnings.append("Non-positive current excluded from logarithmic analysis.")
                        response_value = None
                    else:
                        response_value = float(np.log10(abs(response_value)))
                summary_rows.append(
                    {
                        "file_id": file_id,
                        "file_name": parsed.file_name,
                        "file_hash": parsed.file_hash,
                        "analysis_mode": settings["analysis_mode"],
                        "concentration": concentration,
                        "concentration_unit": unit,
                        "concentration_base_value": base,
                        "scan_rate_V_s": inputs.get("scan_rate_V_s"),
                        "cycle_selection": inputs.get("cycle_selection"),
                        "cycle_number": result.cycle_number or cycle_label,
                        "sweep_direction": result.sweep_direction,
                        "peak_type": result.peak_type,
                        "peak_potential_V": result.peak_potential_V,
                        "raw_peak_current_A": result.raw_current_A,
                        "corrected_peak_current_A": result.corrected_current_A,
                        "response": response_value,
                        "raw_response": response_value,
                        "blank_response": 0.0,
                        "blank_corrected_response": response_value,
                        "display_peak_current": (
                            convert_current(result.corrected_current_A, settings["current_unit"])
                            if result.corrected_current_A is not None and settings["response"] != "Integrated charge"
                            else result.corrected_current_A
                        ),
                        "display_current_unit": "C" if settings["response"] == "Integrated charge" else settings["current_unit"],
                        "fixed_potential_V": result.fixed_potential_V,
                        "integrated_charge_C": result.integrated_charge_C,
                        "included_in_regression": bool(inputs.get("include_in_regression", True)) and (
                            base is not None
                            if settings["analysis_mode"] == "Concentration series"
                            else inputs.get("scan_rate_V_s") is not None
                        ),
                        "warning": "; ".join(dict.fromkeys(warnings)),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    processed = pd.concat(processed_frames, ignore_index=True, sort=False) if processed_frames else pd.DataFrame()
    return summary, processed, top_warnings


def prepare_regression_points(summary: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, list[str], str, str]:
    warnings: list[str] = []
    if summary.empty:
        return pd.DataFrame(), warnings, "", ""
    points = summary.copy()
    points["included"] = points["included_in_regression"].astype(bool)
    if settings["analysis_mode"] == "Concentration series":
        used_units = set(points.loc[points["included"], "concentration_unit"].dropna())
        si_units = set(CONCENTRATION_TO_M)
        target = settings["concentration_x_unit"]
        if not used_units:
            # All concentration values are still blank. This is a missing-input
            # state, not a mixed-unit error.
            points["x"] = np.nan
            x_unit = target
        elif used_units.issubset(si_units):
            points["x"] = [
                convert_concentration(float(value), unit, target)
                if pd.notna(value) and unit in si_units
                else np.nan
                for value, unit in zip(points["concentration"], points["concentration_unit"])
            ]
            x_unit = target
        elif len(used_units) == 1 and next(iter(used_units), None) in ("ppm", "Custom"):
            points["x"] = pd.to_numeric(points["concentration"], errors="coerce")
            selected_unit = next(iter(used_units))
            x_unit = settings["custom_concentration_unit"] if selected_unit == "Custom" else selected_unit
        else:
            points["x"] = np.nan
            points["included"] = False
            x_unit = "mixed"
            warnings.append("SI, ppm, and custom concentration units cannot be mixed in one regression.")
        x_variable = "Concentration"
    else:
        transformed: list[float] = []
        for index, rate in points["scan_rate_V_s"].items():
            try:
                transformed.append(transform_scan_rate(float(rate), settings["scan_x"]))
            except (TypeError, ValueError):
                transformed.append(np.nan)
                points.loc[index, "included"] = False
                warnings.append(f"{points.loc[index, 'file_name']}: invalid scan rate; point excluded.")
        points["x"] = transformed
        x_variable = settings["scan_x"]
        x_unit = {"scan rate, v": "V/s", "square root of scan rate, sqrt(v)": "(V/s)^0.5", "logarithm of scan rate, log10(v)": "log10(V/s)"}[settings["scan_x"]]
    if settings["response"].startswith("log10"):
        points["y"] = points["response"]
        y_unit = "log10(A)"
    elif settings["response"] == "Integrated charge":
        points["y"] = points["response"]
        y_unit = "C"
    else:
        points["y"] = convert_current(points["response"], settings["current_unit"])
        y_unit = settings["current_unit"]
    points.loc[~np.isfinite(points["x"]) | ~np.isfinite(points["y"]), "included"] = False
    if settings["x_range"] is not None:
        low, high = settings["x_range"]
        points.loc[~points["x"].between(low, high, inclusive="both"), "included"] = False
    return points, warnings, x_variable, x_unit + "|" + y_unit


def create_analysis_figure(
    points: pd.DataFrame,
    fit_points: pd.DataFrame,
    result: RegressionResult | None,
    settings: dict[str, Any],
    x_variable: str,
    x_unit: str,
    y_unit: str,
) -> go.Figure:
    figure = go.Figure()
    excluded = points[~points["included"]]
    included = points[points["included"]]
    figure.add_trace(go.Scatter(
        x=included["x"], y=included["y"], mode="markers", name="Included individual points",
        marker={"size": 10, "color": "#1f77b4"}, text=included["file_name"],
        hovertemplate="%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<extra></extra>",
    ))
    if not excluded.empty:
        figure.add_trace(go.Scatter(
            x=excluded["x"], y=excluded["y"], mode="markers", name="Excluded",
            marker={"size": 10, "color": "#999", "symbol": "x"}, text=excluded["file_name"],
            hovertemplate="Excluded · %{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<extra></extra>",
        ))
    if settings["replicate_mode"] != "Show individual points" and not fit_points.empty:
        error = fit_points.get("error", pd.Series(0.0, index=fit_points.index))
        figure.add_trace(go.Scatter(
            x=fit_points["x"], y=fit_points["y"], mode="markers", name=settings["replicate_mode"],
            marker={"size": 12, "color": "#ff7f0e", "symbol": "diamond"},
            error_y={"type": "data", "array": error, "visible": bool(np.any(error > 0))},
        ))
    if result is not None and not fit_points.empty:
        grid = np.linspace(float(fit_points["x"].min()), float(fit_points["x"].max()), 200)
        prediction = result.slope * grid + result.intercept
        figure.add_trace(go.Scatter(x=grid, y=prediction, mode="lines", name="Regression", line={"color": "#d62728"}))
        if settings["confidence_interval"]:
            lower, upper = regression_confidence_band(result, fit_points["x"], grid)
            figure.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[upper, lower[::-1]], fill="toself", fillcolor="rgba(214,39,40,0.12)", line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", name="95% confidence interval"))
        figure.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, text=f"{result.equation}<br>R² = {result.r_squared:.{settings['significant_digits']}g}", bgcolor="rgba(255,255,255,0.85)")
    title = "Calibration curve" if settings["analysis_mode"] == "Concentration series" else "Scan-rate analysis"
    figure.update_layout(title=title, xaxis_title=f"{x_variable} / {x_unit}", yaxis_title=f"{settings['response']} / {y_unit}", template="plotly_white", height=560)
    return figure


def run_full_analysis(settings: dict[str, Any]) -> dict[str, Any]:
    summary, processed, warnings = analysis_rows(settings)
    regression_result: RegressionResult | None = None
    analysis_figure: go.Figure | None = None
    points, point_warnings, x_variable, units = prepare_regression_points(summary, settings)
    warnings.extend(point_warnings)
    if not points.empty and len(points) == len(summary):
        summary.loc[points.index, "included_in_regression"] = points["included"].astype(bool)
    if "|" in units:
        x_unit, y_unit = units.split("|", 1)
    else:
        x_unit, y_unit = units, settings["current_unit"]
    if not points.empty and settings["analysis_mode"] == "Concentration series" and points["concentration"].isna().any():
        warnings.append("Enter a concentration for every file included in the calibration. Missing files remain excluded.")
    included = points[points["included"]].copy() if not points.empty else pd.DataFrame()
    fit_points = aggregate_replicates(included, mode=settings["replicate_mode"]) if not included.empty else included
    if settings["response"] == "Both":
        warnings.append("'Both' displays both peak types but does not fit a single regression. Select one response to regress.")
    elif len(fit_points) >= 2 and fit_points["x"].nunique() >= 2:
        try:
            regression_result = linear_regression(
                fit_points["x"], fit_points["y"],
                through_origin=settings["regression_type"] == "Linear regression through origin",
                analysis_mode=settings["analysis_mode"], x_variable=x_variable, x_unit=x_unit,
                y_variable=settings["response"], y_unit=y_unit,
                significant_digits=settings["significant_digits"], indices=fit_points.index,
            )
        except ValueError as error:
            warnings.append(str(error))
    elif not points.empty and not included.empty:
        warnings.append("Regression requires at least two included points with distinct x values.")
    if not points.empty:
        analysis_figure = create_analysis_figure(points, fit_points, regression_result, settings, x_variable, x_unit, y_unit)
    return {
        "summary": summary,
        "processed": processed,
        "points": points,
        "regression": regression_result,
        "analysis_figure": analysis_figure,
        "warnings": list(dict.fromkeys(warnings)),
        "settings": settings,
    }


require_authentication()
initialize_state()

st.title("CV Calibration & Scan-Rate Analyzer")
st.caption("Upload VersaStudio .par files, inspect CV curves, extract electrochemical responses, and generate calibration or scan-rate plots.")

with st.sidebar:
    st.header("Analysis")
    st.radio("Analysis mode", ["Concentration series", "Scan-rate series"], key="analysis_mode")
    scan_rate_display_unit = st.selectbox("Scan-rate display unit", ["V/s", "mV/s"], key="scan_rate_display_unit_setting")
    current_unit = st.selectbox("Current unit", list(CURRENT_UNIT_FACTORS), index=2, key="current_unit_setting")
    st.divider()
    st.caption("Uploaded files stay in this active session's memory and are not intentionally stored permanently.")

st.subheader("1. Upload .par files")
upload_column, action_column = st.columns([4, 1])
with upload_column:
    uploads = st.file_uploader(
        "VersaStudio files",
        type=["par"],
        accept_multiple_files=True,
        key=f"par_uploader_{st.session_state.uploader_version}",
        help="Limits: 20 MB per file, 30 files, 200 MB per session.",
    )
with action_column:
    st.metric("Parsed files", len(st.session_state.parsed_files))
    if st.button("Reset all", width="stretch"):
        reset_all()
        st.rerun()
process_uploads(uploads)

for error in st.session_state.parse_errors[-10:]:
    st.error(error)

if not st.session_state.parsed_files:
    st.info("Upload one or more `.par` files to begin. Concentration values will remain empty until you enter them manually.")
    st.stop()

remove_names = {
    f"{parsed.file_name} · {parsed.file_hash[:8]}": file_id
    for file_id, parsed in st.session_state.parsed_files.items()
}
with st.expander("Remove uploaded files"):
    remove_selected = st.multiselect("Files to remove", list(remove_names))
    if st.button("Remove selected", disabled=not remove_selected):
        for label in remove_selected:
            file_id = remove_names[label]
            st.session_state.dismissed_ids.add(file_id)
            st.session_state.parsed_files.pop(file_id, None)
            st.session_state.file_inputs.pop(file_id, None)
        st.session_state.analysis_outputs = None
        st.rerun()

st.subheader("2. Review and apply file inputs")
if st.session_state.analysis_mode == "Concentration series":
    st.warning("Concentration is not stored in the .par file. Enter the concentration manually for every file.")

all_cycle_options = ["First cycle", "Last cycle"]
maximum_cycle = max(int(parsed.raw_data["cycle_number"].max()) for parsed in st.session_state.parsed_files.values())
all_cycle_options += [f"Cycle {number}" for number in range(1, maximum_cycle + 1)]
all_cycle_options += ["Last N cycles average", "All cycles separately"]
table_rows = []
scan_factor = 1000.0 if scan_rate_display_unit == "mV/s" else 1.0
for file_id, parsed in st.session_state.parsed_files.items():
    inputs = st.session_state.file_inputs[file_id]
    row = {
        "File ID": file_id,
        "Use": inputs["use"],
        "File name": parsed.file_name,
    }
    if st.session_state.analysis_mode == "Concentration series":
        row["Concentration"] = inputs["concentration"]
        row["Unit"] = inputs["concentration_unit"]
    row.update({
        f"Scan rate ({scan_rate_display_unit})": inputs["scan_rate_V_s"] * scan_factor if inputs["scan_rate_V_s"] is not None else None,
        "Cycle": inputs["cycle_selection"],
        "Include in regression": inputs["include_in_regression"],
        "Parsing status": "Parsed with warning" if parsed.warnings else "Parsed",
    })
    table_rows.append(row)
table = pd.DataFrame(table_rows)
columns: dict[str, Any] = {
    "File ID": None,
    "Use": st.column_config.CheckboxColumn(),
    "File name": st.column_config.TextColumn(disabled=True),
    f"Scan rate ({scan_rate_display_unit})": st.column_config.NumberColumn(min_value=0.0, format="%.8g"),
    "Cycle": st.column_config.SelectboxColumn(options=all_cycle_options),
    "Include in regression": st.column_config.CheckboxColumn(),
    "Parsing status": st.column_config.TextColumn(disabled=True),
}
if st.session_state.analysis_mode == "Concentration series":
    columns["Concentration"] = st.column_config.NumberColumn(min_value=0.0, format="%.8g")
    columns["Unit"] = st.column_config.SelectboxColumn(options=CONCENTRATION_UNITS)
edited = st.data_editor(table, hide_index=True, column_config=columns, disabled=["File ID"], width="stretch")
if st.button("Apply inputs", type="primary"):
    for _, row in edited.iterrows():
        file_id = row["File ID"]
        inputs = st.session_state.file_inputs[file_id]
        inputs["use"] = bool(row["Use"])
        scan = row[f"Scan rate ({scan_rate_display_unit})"]
        inputs["scan_rate_V_s"] = float(scan) / scan_factor if pd.notna(scan) else None
        inputs["cycle_selection"] = str(row["Cycle"])
        inputs["include_in_regression"] = bool(row["Include in regression"])
        if st.session_state.analysis_mode == "Concentration series":
            value = row["Concentration"]
            inputs["concentration"] = float(value) if pd.notna(value) and float(value) >= 0 else None
            inputs["concentration_unit"] = str(row["Unit"])
    st.session_state.analysis_outputs = None
    st.success("Inputs applied. Uploaded files were not reparsed.")

st.subheader("3. Analysis settings")
general_tab, plot_tab = st.tabs(["Response", "Plot"])
with general_tab:
    if st.session_state.analysis_mode == "Concentration series":
        response_options = ["Anodic peak current, Ipa", "Cathodic peak current, Ipc", "Both", "abs(Ipa)", "abs(Ipc)", "Ipa - Ipc", "Ipa - abs(Ipc)", "Fixed-potential current", "Potential interval mean current", "Integrated charge"]
        scan_x = "scan rate, v"
    else:
        response_options = ["Anodic peak current, Ipa", "abs(Ipa)", "Cathodic peak current, Ipc", "abs(Ipc)", "log10(abs(Ipa))", "log10(abs(Ipc))", "Fixed-potential current", "Integrated charge"]
        scan_x = st.selectbox(
            "Scan-rate x axis",
            ["scan rate, v", "square root of scan rate, sqrt(v)", "logarithm of scan rate, log10(v)"],
            key="scan_x_setting",
        )
    response = st.selectbox("Electrochemical response", response_options, key="response_setting")
    st.caption("Set a peak-search potential window before running peak analysis.")
    anodic_column, cathodic_column = st.columns(2)
    with anodic_column:
        st.markdown("Anodic peak")
        a1, a2 = st.columns(2)
        anodic_start = a1.number_input("Window start / V", value=None, key="anodic_start")
        anodic_end = a2.number_input("Window end / V", value=None, key="anodic_end")
        anodic_sweep = st.selectbox("Sweep", ["anodic", "cathodic", "any"], key="anodic_peak_sweep")
    with cathodic_column:
        st.markdown("Cathodic peak")
        c1, c2 = st.columns(2)
        cathodic_start = c1.number_input("Window start / V", value=None, key="cathodic_start")
        cathodic_end = c2.number_input("Window end / V", value=None, key="cathodic_end")
        cathodic_sweep = st.selectbox("Sweep", ["cathodic", "anodic", "any"], key="cathodic_peak_sweep")
    other_sweep = st.selectbox("Sweep for fixed/interval/charge response", ["anodic", "cathodic", "any"], key="other_sweep_setting")
    fixed_potential = st.number_input("Fixed potential / V", value=None, key="fixed_potential_setting")
    i1, i2 = st.columns(2)
    interval_start = i1.number_input("Interval start / V", value=None, key="interval_start_setting")
    interval_end = i2.number_input("Interval end / V", value=None, key="interval_end_setting")
    last_n = st.number_input("N for Last N cycles average", min_value=1, value=2, step=1, key="last_n_setting")
with plot_tab:
    show_legend = st.checkbox("Legend", value=True, key="show_legend_setting")
    show_grid = st.checkbox("Grid", value=True, key="show_grid_setting")
    show_zero_line = st.checkbox("Zero line", value=True, key="show_zero_line_setting")

anodic_window = valid_window(anodic_start, anodic_end)
cathodic_window = valid_window(cathodic_start, cathodic_end)
interval = valid_window(interval_start, interval_end)
snapshot = {
    "analysis_mode": st.session_state.analysis_mode,
    "current_unit": current_unit,
    "response": response,
    "last_n": int(last_n),
    "anodic_window": anodic_window,
    "cathodic_window": cathodic_window,
    "interval": interval,
    "fixed_potential": fixed_potential,
    "anodic_sweep": anodic_sweep,
    "cathodic_sweep": cathodic_sweep,
    "other_sweep": other_sweep,
    "smoothing": "None",
    "smooth_window": 11,
    "smooth_order": 3,
    "smooth_peak": False,
    "smooth_plot": False,
    "baseline": "None",
    "baseline_order": 2,
    "baseline_regions": [],
    "regression_type": "Ordinary least squares",
    "replicate_mode": "Show individual points",
    "confidence_interval": False,
    "significant_digits": 4,
    "x_range": None,
    "scan_x": scan_x,
    "concentration_x_unit": "mM",
    "custom_concentration_unit": "custom",
    "plot_trace": "Raw",
    "show_legend": show_legend,
    "show_grid": show_grid,
    "show_zero_line": show_zero_line,
}

if st.button("Run analysis", type="primary", width="stretch"):
    try:
        st.session_state.analysis_outputs = run_full_analysis(snapshot)
    except (ValueError, TypeError) as error:
        st.error(f"Analysis could not be completed: {error}")

outputs = st.session_state.analysis_outputs
if outputs is None:
    st.info("Apply the file inputs, enter the required response settings, and run the analysis.")
    st.stop()

for warning in outputs["warnings"]:
    st.warning(warning)

summary = outputs["summary"]
processed = outputs["processed"]
st.subheader("4. Results")
highlight_file: str | None = None
if not summary.empty:
    display_summary = summary.copy()
    if response == "Integrated charge":
        raw_label, corrected_label = "Raw charge (C)", "Corrected charge (C)"
        display_summary[raw_label] = display_summary["raw_peak_current_A"]
        display_summary[corrected_label] = display_summary["corrected_peak_current_A"]
    else:
        raw_label, corrected_label = f"Raw current ({current_unit})", f"Corrected current ({current_unit})"
        display_summary[raw_label] = convert_current(display_summary["raw_peak_current_A"], current_unit)
        display_summary[corrected_label] = convert_current(display_summary["corrected_peak_current_A"], current_unit)
    display_columns = ["file_name", "concentration", "concentration_unit", "scan_rate_V_s", "cycle_number", "sweep_direction", "peak_type", "peak_potential_V", raw_label, corrected_label, "display_peak_current", "display_current_unit", "included_in_regression", "warning"]
    event = st.dataframe(display_summary[display_columns], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
    try:
        selected_rows = event.selection.rows
        if selected_rows:
            highlight_file = str(summary.iloc[selected_rows[0]]["file_name"])
    except (AttributeError, KeyError, TypeError):
        pass
else:
    st.info("No response rows were extracted.")

if not processed.empty:
    cv_figure = create_cv_figure(
        processed, summary, current_unit, show_legend, show_grid, show_zero_line,
        "Raw", anodic_window, cathodic_window, highlight_file,
    )
    st.plotly_chart(cv_figure, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "cv_overlay"}})
else:
    cv_figure = None

regression_result: RegressionResult | None = outputs["regression"]
analysis_figure = outputs["analysis_figure"]
if analysis_figure is not None:
    st.plotly_chart(analysis_figure, width="stretch", config={"displaylogo": False})
if regression_result is not None:
    significant_digits = 4
    metrics = st.columns(5)
    metrics[0].metric("Slope", f"{regression_result.slope:.{significant_digits}g}")
    metrics[1].metric("Intercept", f"{regression_result.intercept:.{significant_digits}g}")
    metrics[2].metric("R²", f"{regression_result.r_squared:.{significant_digits}g}")
    metrics[3].metric("RMSE", f"{regression_result.rmse:.{significant_digits}g}")
    metrics[4].metric("n", regression_result.n)
    st.code(regression_result.equation)
    st.dataframe(pd.DataFrame([asdict(regression_result)]).drop(columns=["included_indices"]), hide_index=True, width="stretch")

st.subheader("5. Downloads")
download_columns = st.columns(3)
download_columns[0].download_button("Summary CSV", summary_csv(summary), "summary.csv", "text/csv", width="stretch")
download_columns[1].download_button("Processed CV CSV", processed_csv(processed), "processed_cv_data.csv", "text/csv", width="stretch")
download_columns[2].download_button("Regression CSV", regression_csv(regression_result), "regression_results.csv", "text/csv", width="stretch")
html_columns = st.columns(3)
html_columns[0].download_button("CV plot HTML", figure_html(cv_figure), "cv_overlay.html", "text/html", width="stretch")
html_columns[1].download_button("Analysis plot HTML", figure_html(analysis_figure), "calibration_plot.html", "text/html", width="stretch")
zip_payload = analysis_zip(summary, processed, regression_result, cv_figure, analysis_figure)
html_columns[2].download_button("Complete ZIP", zip_payload, "cv_analysis_results.zip", "application/zip", width="stretch")
with st.expander("Prepare static plot images (PNG/SVG)"):
    st.caption("Static export uses Kaleido and may take a moment on the server. Plotly's camera button is also available above.")
    if cv_figure is not None and st.button("Prepare PNG and SVG"):
        try:
            st.download_button("Download PNG", cv_figure.to_image(format="png", scale=2), "cv_overlay.png", "image/png")
            st.download_button("Download SVG", cv_figure.to_image(format="svg"), "cv_overlay.svg", "image/svg+xml")
        except Exception as error:
            st.error(f"Static image export is unavailable: {error}")
