"""Electrochemical preprocessing and response extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from models.data_models import PeakResult

try:  # scipy is installed in production; the fallback keeps parser tests light.
    from scipy.signal import savgol_filter as _scipy_savgol_filter
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _scipy_savgol_filter = None


CURRENT_UNIT_FACTORS: dict[str, float] = {
    "A": 1.0,
    "mA": 1e3,
    "µA": 1e6,
    "nA": 1e9,
}
CONCENTRATION_TO_M: dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "µM": 1e-6,
    "nM": 1e-9,
}


def convert_current(current_A: float | pd.Series | np.ndarray, unit: str):
    """Convert current stored in amperes to a display unit."""

    if unit not in CURRENT_UNIT_FACTORS:
        raise ValueError(f"Unsupported current unit: {unit}")
    return current_A * CURRENT_UNIT_FACTORS[unit]


def concentration_base_value(value: float | None, unit: str) -> float | None:
    """Return an SI concentration in M, or the unchanged non-SI value."""

    if value is None or not np.isfinite(value) or value < 0:
        return None
    return float(value * CONCENTRATION_TO_M[unit]) if unit in CONCENTRATION_TO_M else float(value)


def convert_concentration(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return float(value)
    if from_unit not in CONCENTRATION_TO_M or to_unit not in CONCENTRATION_TO_M:
        raise ValueError("ppm and custom concentration units cannot be automatically converted.")
    return float(value * CONCENTRATION_TO_M[from_unit] / CONCENTRATION_TO_M[to_unit])


def validate_savgol(window_length: int, polynomial_order: int, point_count: int) -> tuple[int, int]:
    """Validate and normalize Savitzky–Golay settings."""

    window = int(window_length)
    order = int(polynomial_order)
    if window < 3:
        raise ValueError("Savitzky–Golay window length must be at least 3.")
    if window % 2 == 0:
        window += 1
    if window > point_count:
        window = point_count if point_count % 2 else point_count - 1
    if window < 3 or order >= window or order < 0:
        raise ValueError("Polynomial order must be non-negative and smaller than the window.")
    return window, order


def smooth_current(data: pd.DataFrame, window_length: int = 11, polynomial_order: int = 3) -> pd.Series:
    """Smooth current independently within cycle/sweep groups."""

    output = pd.Series(index=data.index, dtype=float)
    group_columns = [column for column in ("cycle_number", "sweep_direction") if column in data]
    grouped = data.groupby(group_columns, sort=False, dropna=False) if group_columns else [(None, data)]
    for _, group in grouped:
        values = group["current_A"].to_numpy(dtype=float)
        try:
            window, order = validate_savgol(window_length, polynomial_order, len(values))
        except ValueError:
            output.loc[group.index] = values
            continue
        if _scipy_savgol_filter is not None:
            smoothed = _scipy_savgol_filter(values, window, order, mode="interp")
        else:  # centered mean fallback; deployment uses scipy.
            smoothed = pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()
        output.loc[group.index] = smoothed
    return output


def _region_mask(potential: pd.Series, regions: Sequence[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=potential.index)
    for start, end in regions:
        low, high = sorted((float(start), float(end)))
        mask |= potential.between(low, high, inclusive="both")
    return mask


def calculate_baseline(
    data: pd.DataFrame,
    method: str = "None",
    polynomial_order: int = 2,
    regions: Sequence[tuple[float, float]] | None = None,
    peak_window: tuple[float, float] | None = None,
    source_column: str = "smoothed_current_A",
) -> pd.Series:
    """Calculate a baseline without modifying raw current."""

    baseline = pd.Series(0.0, index=data.index, dtype=float)
    if method == "None":
        return baseline
    if source_column not in data:
        source_column = "current_A"
    group_columns = [column for column in ("cycle_number", "sweep_direction") if column in data]
    grouped = data.groupby(group_columns, sort=False, dropna=False) if group_columns else [(None, data)]
    for _, group in grouped:
        x = group["potential_V"].astype(float)
        y = group[source_column].astype(float)
        fit_mask = pd.Series(True, index=group.index)
        degree = 1
        if method == "Polynomial baseline":
            degree = max(1, int(polynomial_order))
        elif method == "Manual baseline regions":
            fit_mask = _region_mask(x, regions or [])
            degree = max(1, int(polynomial_order))
        elif method == "Local baseline around peak":
            if peak_window is None:
                continue
            low, high = sorted(peak_window)
            span = max(high - low, np.finfo(float).eps)
            local_regions = [(low - 0.2 * span, low + 0.15 * span), (high - 0.15 * span, high + 0.2 * span)]
            fit_mask = _region_mask(x, local_regions)
            degree = 1
        elif method != "Linear baseline":
            raise ValueError(f"Unknown baseline method: {method}")
        valid = fit_mask & np.isfinite(x) & np.isfinite(y)
        if valid.sum() <= degree:
            continue
        coefficients = np.polyfit(x[valid], y[valid], degree)
        baseline.loc[group.index] = np.polyval(coefficients, x)
    return baseline


def preprocess_cv(
    data: pd.DataFrame,
    smoothing: bool = False,
    smoothing_window: int = 11,
    smoothing_order: int = 3,
    baseline_method: str = "None",
    baseline_order: int = 2,
    baseline_regions: Sequence[tuple[float, float]] | None = None,
    peak_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Create explicit raw, smoothed, baseline, and corrected columns."""

    frame = data.copy()
    frame["raw_current_A"] = frame["current_A"].astype(float)
    frame["smoothed_current_A"] = (
        smooth_current(frame, smoothing_window, smoothing_order)
        if smoothing
        else frame["raw_current_A"]
    )
    frame["baseline_A"] = calculate_baseline(
        frame,
        method=baseline_method,
        polynomial_order=baseline_order,
        regions=baseline_regions,
        peak_window=peak_window,
    )
    frame["corrected_current_A"] = frame["smoothed_current_A"] - frame["baseline_A"]
    return frame


def _sweep_subset(data: pd.DataFrame, sweep_direction: str | None) -> pd.DataFrame:
    if not sweep_direction or sweep_direction == "any" or "sweep_direction" not in data:
        return data
    return data[data["sweep_direction"] == sweep_direction]


def find_peak(
    data: pd.DataFrame,
    peak_type: Literal["anodic", "cathodic"],
    potential_window: tuple[float, float],
    sweep_direction: str | None = None,
    search_column: str = "corrected_current_A",
) -> PeakResult:
    """Find a peak only inside the user-supplied potential window."""

    if potential_window is None:
        raise ValueError("Set a peak-search potential window before running peak analysis.")
    low, high = sorted((float(potential_window[0]), float(potential_window[1])))
    subset = _sweep_subset(data, sweep_direction)
    subset = subset[subset["potential_V"].between(low, high, inclusive="both")]
    subset = subset[np.isfinite(subset["potential_V"]) & np.isfinite(subset[search_column])]
    if subset.empty:
        return PeakResult(
            peak_type=peak_type,
            peak_potential_V=None,
            raw_current_A=None,
            corrected_current_A=None,
            response_A=None,
            sweep_direction=sweep_direction or "any",
            warnings=["No usable points lie inside the peak-search window."],
        )
    index = subset[search_column].idxmax() if peak_type == "anodic" else subset[search_column].idxmin()
    row = subset.loc[index]
    warnings: list[str] = []
    ordered = subset.sort_values("potential_V", kind="stable")
    if index in (ordered.index[0], ordered.index[-1]):
        warnings.append("Detected peak lies on the search-window boundary. Review the peak window.")
    raw = float(row.get("raw_current_A", row["current_A"]))
    corrected = float(row.get("corrected_current_A", row[search_column]))
    return PeakResult(
        peak_type=peak_type,
        peak_potential_V=float(row["potential_V"]),
        raw_current_A=raw,
        corrected_current_A=corrected,
        response_A=corrected,
        cycle_number=int(row["cycle_number"]) if "cycle_number" in row and pd.notna(row["cycle_number"]) else None,
        sweep_direction=str(row.get("sweep_direction", sweep_direction or "any")),
        warnings=warnings,
    )


def current_at_potential(
    data: pd.DataFrame,
    potential_V: float,
    sweep_direction: str | None = None,
    current_column: str = "corrected_current_A",
) -> PeakResult:
    """Linearly interpolate current at an exact potential on one sweep."""

    subset = _sweep_subset(data, sweep_direction)
    subset = subset[["potential_V", current_column, "raw_current_A"]].dropna()
    if subset.empty:
        raise ValueError("No data are available for fixed-potential interpolation.")
    grouped = subset.groupby("potential_V", as_index=False).mean(numeric_only=True).sort_values("potential_V")
    x = grouped["potential_V"].to_numpy(dtype=float)
    if potential_V < x.min() or potential_V > x.max():
        raise ValueError("Fixed potential lies outside the selected sweep range.")
    corrected = float(np.interp(potential_V, x, grouped[current_column].to_numpy(dtype=float)))
    raw = float(np.interp(potential_V, x, grouped["raw_current_A"].to_numpy(dtype=float)))
    return PeakResult(
        peak_type="fixed-potential current",
        peak_potential_V=float(potential_V),
        raw_current_A=raw,
        corrected_current_A=corrected,
        response_A=corrected,
        sweep_direction=sweep_direction or "any",
        fixed_potential_V=float(potential_V),
    )


def interval_mean_current(
    data: pd.DataFrame,
    potential_window: tuple[float, float],
    sweep_direction: str | None = None,
    current_column: str = "corrected_current_A",
) -> PeakResult:
    low, high = sorted(potential_window)
    subset = _sweep_subset(data, sweep_direction)
    subset = subset[subset["potential_V"].between(low, high, inclusive="both")]
    if subset.empty:
        raise ValueError("No data lie in the potential interval.")
    corrected = float(subset[current_column].mean())
    raw = float(subset["raw_current_A"].mean())
    return PeakResult(
        peak_type="potential interval mean current",
        peak_potential_V=float(subset["potential_V"].mean()),
        raw_current_A=raw,
        corrected_current_A=corrected,
        response_A=corrected,
        sweep_direction=sweep_direction or "any",
    )


def integrate_charge(
    data: pd.DataFrame,
    potential_window: tuple[float, float],
    sweep_direction: str | None = None,
    current_column: str = "corrected_current_A",
) -> PeakResult:
    """Integrate I dt with the elapsed-time column and trapezoidal integration."""

    low, high = sorted(potential_window)
    subset = _sweep_subset(data, sweep_direction)
    subset = subset[subset["potential_V"].between(low, high, inclusive="both")]
    if "elapsed_time_s" not in subset or subset["elapsed_time_s"].notna().sum() < 2:
        raise ValueError("Integrated charge requires at least two elapsed-time values.")
    subset = subset.dropna(subset=["elapsed_time_s", current_column]).sort_values("elapsed_time_s")
    if len(subset) < 2:
        raise ValueError("Integrated charge requires at least two valid points.")
    charge = float(np.trapezoid(subset[current_column], subset["elapsed_time_s"]))
    raw_charge = float(np.trapezoid(subset["raw_current_A"], subset["elapsed_time_s"]))
    return PeakResult(
        peak_type="integrated charge",
        peak_potential_V=None,
        raw_current_A=raw_charge,
        corrected_current_A=charge,
        response_A=charge,
        sweep_direction=sweep_direction or "any",
        integrated_charge_C=charge,
    )


def combine_peak_response(response: str, anodic: PeakResult, cathodic: PeakResult) -> float:
    """Combine separately detected peaks into a calibration response."""

    ipa = anodic.response_A
    ipc = cathodic.response_A
    if ipa is None or ipc is None:
        raise ValueError("Both anodic and cathodic peaks are required for this response.")
    mapping = {
        "Ipa - Ipc": ipa - ipc,
        "Ipa - abs(Ipc)": ipa - abs(ipc),
    }
    if response not in mapping:
        raise ValueError(f"Unsupported combined response: {response}")
    return float(mapping[response])


def transform_scan_rate(scan_rate_V_s: float, variable: str) -> float:
    if not np.isfinite(scan_rate_V_s) or scan_rate_V_s <= 0:
        raise ValueError("Scan rate must be greater than zero for this transform.")
    if variable == "scan rate, v":
        return float(scan_rate_V_s)
    if variable == "square root of scan rate, sqrt(v)":
        return float(np.sqrt(scan_rate_V_s))
    if variable == "logarithm of scan rate, log10(v)":
        return float(np.log10(scan_rate_V_s))
    raise ValueError(f"Unknown scan-rate x variable: {variable}")
