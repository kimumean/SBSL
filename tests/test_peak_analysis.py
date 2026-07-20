from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import asdict
from streamlit.testing.v1 import AppTest

from core.cv_analysis import (
    concentration_base_value,
    current_at_potential,
    find_peak,
    integrate_charge,
    preprocess_cv,
    transform_scan_rate,
)
from core.cycle_detection import annotate_cycles, select_cycles
from core.par_parser import parse_par_bytes
from models.data_models import FileInput, preserve_file_input


def _last_cycle(par_bytes: bytes) -> pd.DataFrame:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    detected = annotate_cycles(parsed.raw_data, parsed.metadata)
    selected = select_cycles(detected, "Last cycle")[0][1]
    return preprocess_cv(selected)


def test_peak_is_detected_inside_window(par_bytes: bytes) -> None:
    data = _last_cycle(par_bytes)
    result = find_peak(data, "anodic", (0.0, 0.3), "anodic")
    assert result.peak_potential_V == pytest.approx(0.2)
    assert result.response_A == pytest.approx(5.5e-6)


def test_cathodic_peak_uses_minimum(par_bytes: bytes) -> None:
    data = _last_cycle(par_bytes)
    result = find_peak(data, "cathodic", (-0.1, 0.3), "cathodic")
    assert result.peak_potential_V == pytest.approx(0.0)
    assert result.response_A == pytest.approx(-4.5e-6)


def test_peak_boundary_warning(par_bytes: bytes) -> None:
    data = _last_cycle(par_bytes)
    result = find_peak(data, "anodic", (0.19, 0.21), "anodic")
    assert any("boundary" in warning for warning in result.warnings)


def test_fixed_potential_linear_interpolation(par_bytes: bytes) -> None:
    data = _last_cycle(par_bytes)
    result = current_at_potential(data, 0.1, "anodic")
    assert result.response_A == pytest.approx(3.35e-6)


def test_integrated_charge_uses_elapsed_time() -> None:
    data = pd.DataFrame(
        {
            "potential_V": [0.0, 0.1, 0.2],
            "current_A": [1e-6, 1e-6, 1e-6],
            "elapsed_time_s": [0.0, 1.0, 2.0],
            "cycle_number": [1, 1, 1],
            "sweep_direction": ["anodic"] * 3,
        }
    )
    result = integrate_charge(preprocess_cv(data), (0.0, 0.2), "anodic")
    assert result.integrated_charge_C == pytest.approx(2e-6)


def test_sqrt_scan_rate() -> None:
    assert transform_scan_rate(0.04, "square root of scan rate, sqrt(v)") == pytest.approx(0.2)


def test_concentration_default_is_null_and_manual_value_is_used() -> None:
    inputs = FileInput()
    assert inputs.concentration is None
    inputs.concentration = 2.5
    assert concentration_base_value(inputs.concentration, inputs.concentration_unit) == pytest.approx(0.0025)


def test_nonpositive_log_scan_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        transform_scan_rate(0.0, "logarithm of scan rate, log10(v)")


def test_new_file_does_not_overwrite_manual_session_input() -> None:
    stored: dict[str, dict[str, object]] = {}
    first = preserve_file_input(stored, "first", FileInput(scan_rate_V_s=0.05))
    first["concentration"] = 2.5
    preserve_file_input(stored, "second", FileInput(scan_rate_V_s=0.10))
    preserve_file_input(stored, "first", FileInput(scan_rate_V_s=9.99))
    assert stored["first"]["concentration"] == 2.5
    assert stored["first"]["scan_rate_V_s"] == 0.05


def test_app_without_peak_window_keeps_cv_plot_and_shows_warning(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    parsed.raw_data = annotate_cycles(parsed.raw_data, parsed.metadata)
    app = AppTest.from_file("app.py")
    app.session_state["parsed_files"] = {parsed.file_id: parsed}
    app.session_state["file_inputs"] = {
        parsed.file_id: asdict(FileInput(concentration=1.0, scan_rate_V_s=0.05))
    }
    app.session_state["analysis_mode"] = "Concentration series"
    app.run(timeout=30)
    next(button for button in app.button if button.label == "Run analysis").click().run(timeout=30)
    assert not app.exception
    outputs = app.session_state["analysis_outputs"]
    assert outputs["summary"].empty
    assert not outputs["processed"].empty
    assert any("peak-search potential window" in warning for warning in outputs["warnings"])


def test_blank_concentrations_do_not_raise_false_mixed_unit_warning(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    parsed.raw_data = annotate_cycles(parsed.raw_data, parsed.metadata)
    app = AppTest.from_file("app.py")
    app.session_state["parsed_files"] = {parsed.file_id: parsed}
    app.session_state["file_inputs"] = {
        parsed.file_id: asdict(FileInput(scan_rate_V_s=0.05))
    }
    app.session_state["analysis_mode"] = "Concentration series"
    app.run(timeout=30)
    app.number_input(key="anodic_start").set_value(0.0)
    app.number_input(key="anodic_end").set_value(0.3)
    next(button for button in app.button if button.label == "Run analysis").click().run(timeout=30)
    assert not app.exception
    warnings = app.session_state["analysis_outputs"]["warnings"]
    assert any("Enter a concentration" in warning for warning in warnings)
    assert not any("cannot be mixed" in warning for warning in warnings)
    assert not any("distinct x values" in warning for warning in warnings)
