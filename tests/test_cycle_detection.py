from __future__ import annotations

import numpy as np
import pandas as pd

from core.cycle_detection import annotate_cycles, select_cycles
from core.par_parser import parse_par_bytes


def test_anodic_and_cathodic_sweeps_are_separated(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    detected = annotate_cycles(parsed.raw_data, parsed.metadata)
    assert set(detected["sweep_direction"]) == {"anodic", "cathodic"}
    assert set(detected.loc[detected["segment_number"] == 1, "sweep_direction"]) == {"anodic"}
    assert set(detected.loc[detected["segment_number"] == 2, "sweep_direction"]) == {"cathodic"}


def test_cycle_detection_uses_turning_directions(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    detected = annotate_cycles(parsed.raw_data, parsed.metadata)
    assert detected["cycle_number"].max() == 2
    assert detected.attrs["declared_cycles"] == 2


def test_last_cycle_selection(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    detected = annotate_cycles(parsed.raw_data, parsed.metadata)
    label, selected = select_cycles(detected, "Last cycle")[0]
    assert label == "Cycle 2"
    assert selected["cycle_number"].nunique() == 1
    assert selected["cycle_number"].iloc[0] == 2


def test_all_cycles_returns_separate_traces(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    detected = annotate_cycles(parsed.raw_data, parsed.metadata)
    traces = select_cycles(detected, "All cycles separately")
    assert [label for label, _ in traces] == ["Cycle 1", "Cycle 2"]


def test_repeated_switch_point_is_not_a_false_cycle() -> None:
    data = pd.DataFrame(
        {
            "potential_V": [-0.2, 0.0, 0.2, 0.2, 0.0, -0.2],
            "current_A": np.arange(6, dtype=float),
            "segment_number": [1, 1, 1, 2, 2, 2],
        }
    )
    detected = annotate_cycles(data)
    assert detected["cycle_number"].max() == 1
