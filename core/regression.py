"""Linear regression, confidence intervals, and replicate aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from models.data_models import RegressionResult

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


def linear_regression(
    x: Iterable[float],
    y: Iterable[float],
    *,
    through_origin: bool = False,
    analysis_mode: str = "Concentration series",
    x_variable: str = "concentration",
    x_unit: str = "",
    y_variable: str = "response",
    y_unit: str = "A",
    significant_digits: int = 4,
    indices: Iterable[int] | None = None,
) -> RegressionResult:
    """Fit OLS or a regression constrained through the origin."""

    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape.")
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[valid], y_values[valid]
    n = len(x_values)
    parameters = 1 if through_origin else 2
    if n < max(2, parameters):
        raise ValueError("At least two valid observations are required for regression.")
    if np.unique(x_values).size < 2:
        raise ValueError("Regression requires at least two distinct x values.")

    if through_origin:
        denominator = float(np.dot(x_values, x_values))
        if denominator == 0:
            raise ValueError("A through-origin fit requires at least one non-zero x value.")
        slope = float(np.dot(x_values, y_values) / denominator)
        intercept = 0.0
    else:
        design = np.column_stack((x_values, np.ones(n)))
        slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
        slope, intercept = float(slope), float(intercept)

    predicted = slope * x_values + intercept
    residuals = y_values - predicted
    sse = float(np.sum(residuals**2))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    if through_origin:
        total = float(np.sum(y_values**2))
    else:
        total = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r_squared = float(1.0 - sse / total) if total > 0 else (1.0 if sse == 0 else float("nan"))
    dof = n - parameters
    adjusted = (
        float(1 - (1 - r_squared) * (n - 1) / dof)
        if dof > 0 and np.isfinite(r_squared)
        else float("nan")
    )
    residual_variance = sse / dof if dof > 0 else 0.0
    if through_origin:
        slope_se = float(np.sqrt(residual_variance / np.sum(x_values**2)))
        intercept_se = 0.0
    else:
        centered = float(np.sum((x_values - np.mean(x_values)) ** 2))
        slope_se = float(np.sqrt(residual_variance / centered)) if centered > 0 else float("nan")
        intercept_se = float(
            np.sqrt(residual_variance * (1 / n + np.mean(x_values) ** 2 / centered))
        ) if centered > 0 else float("nan")
    if slope_se > 0 and np.isfinite(slope_se):
        statistic = abs(slope / slope_se)
        p_value = float(2 * stats.t.sf(statistic, dof)) if stats is not None and dof > 0 else float(math.erfc(statistic / math.sqrt(2)))
    else:
        p_value = 0.0 if slope != 0 else 1.0
    spec = f".{max(1, int(significant_digits))}g"
    if through_origin:
        equation = f"{y_variable} ({y_unit}) = {slope:{spec}} × {x_variable} ({x_unit})"
    else:
        sign = "+" if intercept >= 0 else "−"
        equation = f"{y_variable} ({y_unit}) = {slope:{spec}} × {x_variable} ({x_unit}) {sign} {abs(intercept):{spec}}"
    included = list(indices) if indices is not None else list(range(n))
    if len(included) != len(valid):
        included = list(range(n))
    else:
        included = [index for index, keep in zip(included, valid) if keep]
    return RegressionResult(
        analysis_mode=analysis_mode,
        x_variable=x_variable,
        x_unit=x_unit,
        y_variable=y_variable,
        y_unit=y_unit,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        slope_standard_error=slope_se,
        intercept_standard_error=intercept_se,
        p_value=p_value,
        rmse=rmse,
        n=n,
        equation=equation,
        through_origin=through_origin,
        included_indices=included,
    )


def regression_confidence_band(
    result: RegressionResult,
    x_observed: Iterable[float],
    x_grid: Iterable[float],
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return approximate pointwise confidence bounds for the fitted mean."""

    observed = np.asarray(list(x_observed), dtype=float)
    grid = np.asarray(list(x_grid), dtype=float)
    predicted = result.slope * grid + result.intercept
    dof = result.n - (1 if result.through_origin else 2)
    if dof <= 0:
        return predicted, predicted
    if stats is not None:
        critical = float(stats.t.ppf((1 + confidence) / 2, dof))
    else:
        critical = 1.96
    residual_sd = result.rmse * math.sqrt(result.n / dof)
    if result.through_origin:
        se = residual_sd * np.sqrt(grid**2 / np.sum(observed**2))
    else:
        centered = np.sum((observed - np.mean(observed)) ** 2)
        se = residual_sd * np.sqrt(1 / result.n + (grid - np.mean(observed)) ** 2 / centered)
    return predicted - critical * se, predicted + critical * se


def aggregate_replicates(
    data: pd.DataFrame,
    x_column: str = "x",
    y_column: str = "y",
    mode: str = "Show individual points",
) -> pd.DataFrame:
    """Aggregate replicate responses while retaining n, SD, and SE."""

    if mode == "Show individual points":
        result = data.copy()
        result["mean"] = result[y_column]
        result["std"] = 0.0
        result["sem"] = 0.0
        result["n"] = 1
        return result
    grouped = data.groupby(x_column, dropna=False)[y_column].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"count": "n"})
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"])
    grouped[y_column] = grouped["mean"]
    if mode == "Mean only":
        grouped["error"] = 0.0
    elif mode == "Mean ± standard deviation":
        grouped["error"] = grouped["std"]
    elif mode == "Mean ± standard error":
        grouped["error"] = grouped["sem"]
    else:
        raise ValueError(f"Unknown replicate mode: {mode}")
    return grouped


def apply_blank_correction(
    data: pd.DataFrame,
    mode: str,
    *,
    concentration_column: str = "concentration_base_value",
    response_column: str = "response",
    blank_file_ids: set[str] | None = None,
    manual_blank: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Apply one blank value while preserving the raw response."""

    result = data.copy()
    result["raw_response"] = result[response_column]
    if mode == "No blank correction":
        blank = 0.0
    elif mode == "Use concentration 0 data":
        values = result.loc[result[concentration_column] == 0, response_column].dropna()
        if values.empty:
            raise ValueError("No concentration-0 response is available for blank correction.")
        blank = float(values.mean())
    elif mode == "Use selected files":
        values = result.loc[result["file_id"].isin(blank_file_ids or set()), response_column].dropna()
        if values.empty:
            raise ValueError("No selected blank-file response is available.")
        blank = float(values.mean())
    elif mode == "Enter blank current manually":
        if manual_blank is None or not np.isfinite(manual_blank):
            raise ValueError("Enter a finite manual blank current.")
        blank = float(manual_blank)
    else:
        raise ValueError(f"Unknown blank correction mode: {mode}")
    result[response_column] = result["raw_response"] - blank
    result["blank_response"] = blank
    return result, blank
