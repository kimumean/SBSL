"""Cycle and sweep annotation for cyclic voltammetry data."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _stable_direction(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=object)
    delta = np.diff(values, prepend=values[0])
    scale = max(float(np.nanmax(values) - np.nanmin(values)), 1.0)
    tolerance = max(scale * 1e-8, 1e-12)
    direction = np.full(len(values), "hold", dtype=object)
    direction[delta > tolerance] = "anodic"
    direction[delta < -tolerance] = "cathodic"

    # A single repeated switching-potential point belongs to the surrounding
    # sweep for plotting, while a sustained plateau remains a hold.
    for i in range(len(direction)):
        if direction[i] != "hold":
            continue
        previous = next((direction[j] for j in range(i - 1, -1, -1) if direction[j] != "hold"), None)
        following = next((direction[j] for j in range(i + 1, len(direction)) if direction[j] != "hold"), None)
        if previous == following and previous is not None:
            direction[i] = previous
        elif i == 0 and following is not None:
            direction[i] = following
    return direction


def _segment_directions(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series("unknown", index=frame.index, dtype="object")
    for _, group in frame.groupby("segment_number", sort=False):
        potential = group["potential_V"].to_numpy(dtype=float)
        point_directions = _stable_direction(potential)
        non_hold = point_directions[point_directions != "hold"]
        if len(non_hold):
            values, counts = np.unique(non_hold, return_counts=True)
            dominant = str(values[int(np.argmax(counts))])
            # Segment structure has priority when the segment is monotonic.
            if float(np.max(counts)) / len(non_hold) >= 0.8:
                point_directions[point_directions != "hold"] = dominant
        result.loc[group.index] = point_directions
    return result


def annotate_cycles(data: pd.DataFrame, metadata: dict[str, object] | None = None) -> pd.DataFrame:
    """Assign ``cycle_number``, ``segment_number``, and ``sweep_direction``.

    Segment boundaries are respected first. Potential direction is then used
    inside each segment and as the fallback for files that place multiple
    turning points inside a single data block. A new cycle starts when the
    initial scan direction returns after the opposite scan has occurred.
    """

    required = {"potential_V", "current_A"}
    if not required.issubset(data.columns):
        raise ValueError(f"Cycle detection requires columns: {sorted(required)}")
    frame = data.copy().reset_index(drop=True)
    if "segment_number" not in frame:
        frame["segment_number"] = 1
    frame["sweep_direction"] = _segment_directions(frame)

    non_hold = frame.loc[frame["sweep_direction"].isin(["anodic", "cathodic"]), "sweep_direction"]
    if non_hold.empty:
        frame["cycle_number"] = 1
        frame["sweep_direction"] = "unknown"
        return frame

    initial = str(non_hold.iloc[0])
    opposite_seen = False
    cycle = 1
    previous_scan: str | None = None
    cycles: list[int] = []
    for direction in frame["sweep_direction"].astype(str):
        if direction in ("hold", "unknown"):
            cycles.append(cycle)
            continue
        if previous_scan is not None and direction != previous_scan:
            if direction != initial:
                opposite_seen = True
            elif opposite_seen:
                cycle += 1
                opposite_seen = False
        cycles.append(cycle)
        previous_scan = direction
    frame["cycle_number"] = cycles

    declared = None
    if metadata:
        for key, value in metadata.items():
            if re.sub(r"\s+", "", str(key)).casefold() == "cycles":
                try:
                    declared = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    declared = None
                break
    # Metadata is a consistency hint, not a reason to evenly split points.
    frame.attrs["declared_cycles"] = declared
    frame.attrs["detected_cycles"] = int(frame["cycle_number"].max())
    return frame


def cycle_options(data: pd.DataFrame) -> list[str]:
    maximum = int(data["cycle_number"].max()) if not data.empty else 1
    return ["First cycle", "Last cycle", *(f"Cycle {n}" for n in range(1, maximum + 1)), "Last N cycles average", "All cycles separately"]


def select_cycles(data: pd.DataFrame, selection: str, last_n: int = 2) -> list[tuple[str, pd.DataFrame]]:
    """Return one or more traces for a UI cycle-selection label."""

    if data.empty:
        return []
    cycles = sorted(int(value) for value in data["cycle_number"].dropna().unique())
    if not cycles:
        return [("Cycle 1", data.copy())]
    if selection == "First cycle":
        chosen = cycles[0]
        return [(f"Cycle {chosen}", data[data["cycle_number"] == chosen].copy())]
    if selection == "Last cycle":
        chosen = cycles[-1]
        return [(f"Cycle {chosen}", data[data["cycle_number"] == chosen].copy())]
    if selection == "All cycles separately":
        return [(f"Cycle {cycle}", data[data["cycle_number"] == cycle].copy()) for cycle in cycles]
    match = re.fullmatch(r"Cycle\s+(\d+)", selection)
    if match:
        chosen = int(match.group(1))
        subset = data[data["cycle_number"] == chosen].copy()
        return [(f"Cycle {chosen}", subset)] if not subset.empty else []
    if selection == "Last N cycles average":
        selected = cycles[-max(1, int(last_n)) :]
        subset = data[data["cycle_number"].isin(selected)].copy()
        averaged = average_cycles(subset)
        return [(f"Mean cycles {selected[0]}–{selected[-1]}", averaged)]
    raise ValueError(f"Unknown cycle selection: {selection}")


def average_cycles(data: pd.DataFrame) -> pd.DataFrame:
    """Average cycles point-by-point within each sweep using normalized order."""

    if data.empty or data["cycle_number"].nunique() <= 1:
        return data.copy()
    numeric = [
        column
        for column in ("potential_V", "current_A", "elapsed_time_s", "e_applied_V")
        if column in data
    ]
    pieces: list[pd.DataFrame] = []
    for direction, sweep in data.groupby("sweep_direction", sort=False):
        cycle_groups = [group.reset_index(drop=True) for _, group in sweep.groupby("cycle_number", sort=True)]
        minimum = min(len(group) for group in cycle_groups)
        if minimum < 2:
            continue
        grid = np.linspace(0.0, 1.0, minimum)
        averaged: dict[str, np.ndarray] = {}
        for column in numeric:
            interpolated = []
            for group in cycle_groups:
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
                source = np.linspace(0.0, 1.0, len(values))
                valid = np.isfinite(values)
                if valid.sum() >= 2:
                    interpolated.append(np.interp(grid, source[valid], values[valid]))
            if interpolated:
                averaged[column] = np.mean(interpolated, axis=0)
        piece = pd.DataFrame(averaged)
        piece["sweep_direction"] = direction
        piece["cycle_number"] = int(max(data["cycle_number"]))
        piece["segment_number"] = -1
        piece["point_number"] = range(minimum)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else data.copy()
