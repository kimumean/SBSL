"""Typed data containers shared by parsing, analysis, and the web UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class CVFileData:
    """One in-memory VersaStudio file and its parsed contents."""

    file_id: str
    file_name: str
    metadata: dict[str, Any]
    raw_data: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    file_hash: str = ""
    file_size: int = 0
    encoding: str = "unknown"


@dataclass(slots=True)
class FileInput:
    """Editable per-file values; concentration intentionally starts empty."""

    use: bool = True
    concentration: float | None = None
    concentration_unit: str = "mM"
    scan_rate_V_s: float | None = None
    cycle_selection: str = "Last cycle"
    include_in_regression: bool = True


@dataclass(slots=True)
class PeakResult:
    """A peak or other electrochemical response extracted from one trace."""

    peak_type: str
    peak_potential_V: float | None
    raw_current_A: float | None
    corrected_current_A: float | None
    response_A: float | None
    cycle_number: int | None = None
    sweep_direction: str = "unknown"
    fixed_potential_V: float | None = None
    integrated_charge_C: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegressionResult:
    """Linear-regression statistics ready for display or export."""

    analysis_mode: str
    x_variable: str
    x_unit: str
    y_variable: str
    y_unit: str
    slope: float
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    slope_standard_error: float
    intercept_standard_error: float
    p_value: float
    rmse: float
    n: int
    equation: str
    through_origin: bool = False
    included_indices: list[int] = field(default_factory=list)


def preserve_file_input(
    inputs: dict[str, dict[str, Any]], file_id: str, default: FileInput
) -> dict[str, Any]:
    """Add a new file default without overwriting existing manual values."""

    if file_id not in inputs:
        inputs[file_id] = {
            "use": default.use,
            "concentration": default.concentration,
            "concentration_unit": default.concentration_unit,
            "scan_rate_V_s": default.scan_rate_V_s,
            "cycle_selection": default.cycle_selection,
            "include_in_regression": default.include_in_regression,
        }
    return inputs[file_id]
