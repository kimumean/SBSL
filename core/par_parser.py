"""Parser for text-based VersaStudio ``.par`` files.

The format is a structured text document rather than a CSV file.  Global
metadata and each ``<SegmentN>`` block are therefore parsed before individual
comma-separated data records are decoded.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from models.data_models import CVFileData


class ParParseError(ValueError):
    """Raised when a file cannot be recognized as a usable VersaStudio file."""


ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp949", "latin-1")
SEGMENT_RE = re.compile(
    r"<Segment\s*(\d+)\s*>(.*?)</Segment\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
META_RE = re.compile(r"^\s*([^=\r\n]+?)\s*=\s*(.*?)\s*$")
REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "potential_V": ("E(V)", "E (V)", "Potential(V)"),
    "current_A": ("I(A)", "I (A)", "Current(A)"),
}


def decode_par_bytes(contents: bytes) -> tuple[str, str]:
    """Decode uploaded bytes without writing them to disk."""

    if not contents:
        raise ParParseError("The uploaded file is empty.")
    for encoding in ENCODINGS:
        try:
            return contents.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ParParseError("The file could not be decoded with a supported encoding.")


def _coerce_metadata(value: str) -> Any:
    cleaned = value.strip().strip('"')
    if not cleaned:
        return ""
    try:
        if re.fullmatch(r"[-+]?\d+", cleaned):
            return int(cleaned)
        if re.fullmatch(
            r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?", cleaned
        ):
            return float(cleaned)
    except ValueError:
        pass
    return cleaned


def extract_metadata(text: str) -> dict[str, Any]:
    """Extract global metadata, excluding the bodies of segment blocks."""

    global_text = SEGMENT_RE.sub("", text)
    metadata: dict[str, Any] = {}
    for line in global_text.splitlines():
        match = META_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        metadata[key.strip()] = _coerce_metadata(value)
    return metadata


def _unique_columns(columns: Iterable[str]) -> list[str]:
    result: list[str] = []
    counts: dict[str, int] = {}
    for index, raw in enumerate(columns):
        name = raw.strip().strip('"') or f"unnamed_{index}"
        count = counts.get(name, 0)
        counts[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count + 1}")
    return result


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {re.sub(r"\s+", "", col).lower(): col for col in columns}
    for alias in aliases:
        found = normalized.get(re.sub(r"\s+", "", alias).lower())
        if found is not None:
            return found
    return None


def extract_segment_blocks(text: str) -> list[tuple[int, str]]:
    """Return ordered ``(segment number, block body)`` pairs."""

    return [(int(number), body) for number, body in SEGMENT_RE.findall(text)]


def _parse_segment(segment_number: int, body: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    lines = body.splitlines()
    definition_index: int | None = None
    columns: list[str] = []
    for index, line in enumerate(lines):
        if line.lstrip().lower().startswith("definition="):
            definition_index = index
            definition = line.split("=", 1)[1]
            columns = _unique_columns(next(csv.reader([definition], skipinitialspace=True)))
            break
    if definition_index is None or not columns:
        return pd.DataFrame(), [f"Segment {segment_number}: missing Definition row."]

    potential_col = _find_column(columns, REQUIRED_ALIASES["potential_V"])
    current_col = _find_column(columns, REQUIRED_ALIASES["current_A"])
    if potential_col is None or current_col is None:
        return pd.DataFrame(), [
            f"Segment {segment_number}: Definition does not contain E(V) and I(A)."
        ]

    rows: list[list[str]] = []
    rejected: dict[str, int] = {}
    for line_number, line in enumerate(lines[definition_index + 1 :], definition_index + 2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" in stripped:
            continue
        try:
            values = next(csv.reader([stripped], skipinitialspace=True))
        except csv.Error:
            rejected["invalid CSV syntax"] = rejected.get("invalid CSV syntax", 0) + 1
            continue
        if len(values) < len(columns):
            rejected["too few fields"] = rejected.get("too few fields", 0) + 1
            continue
        if len(values) > len(columns):
            # Some VersaStudio versions append empty fields. Preserve meaningful
            # columns and ignore the extra tail instead of failing the file.
            values = values[: len(columns)]
        try:
            float(values[columns.index(potential_col)].strip())
            float(values[columns.index(current_col)].strip())
        except (ValueError, IndexError):
            rejected["non-numeric E(V) or I(A)"] = rejected.get(
                "non-numeric E(V) or I(A)", 0
            ) + 1
            continue
        rows.append([value.strip() for value in values])

    if rejected:
        details = ", ".join(f"{count} {reason}" for reason, count in rejected.items())
        warnings.append(f"Segment {segment_number}: skipped {sum(rejected.values())} row(s): {details}.")
    if not rows:
        warnings.append(f"Segment {segment_number}: no usable data rows.")
        return pd.DataFrame(), warnings

    frame = pd.DataFrame(rows, columns=columns)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted

    frame["potential_V"] = pd.to_numeric(frame[potential_col], errors="coerce")
    frame["current_A"] = pd.to_numeric(frame[current_col], errors="coerce")
    elapsed_col = _find_column(frame.columns, ("Elapsed Time(s)", "Time(s)"))
    point_col = _find_column(frame.columns, ("Point #", "Point#", "Point"))
    source_segment_col = _find_column(frame.columns, ("Segment #", "Segment#"))
    applied_col = _find_column(frame.columns, ("E Applied(V)", "E Applied (V)"))
    frame["elapsed_time_s"] = (
        pd.to_numeric(frame[elapsed_col], errors="coerce")
        if elapsed_col
        else pd.Series(float("nan"), index=frame.index)
    )
    frame["point_number"] = (
        pd.to_numeric(frame[point_col], errors="coerce")
        if point_col
        else pd.Series(range(len(frame)), index=frame.index, dtype=float)
    )
    frame["source_segment_number"] = (
        pd.to_numeric(frame[source_segment_col], errors="coerce")
        if source_segment_col
        else segment_number
    )
    frame["e_applied_V"] = (
        pd.to_numeric(frame[applied_col], errors="coerce")
        if applied_col
        else pd.Series(float("nan"), index=frame.index)
    )
    frame["segment_number"] = segment_number
    frame["_row_order"] = range(len(frame))
    return frame, warnings


def metadata_value(metadata: dict[str, Any], key: str) -> Any | None:
    """Find a metadata value with whitespace/case-insensitive matching."""

    wanted = re.sub(r"\s+", "", key).casefold()
    for existing_key, value in metadata.items():
        if re.sub(r"\s+", "", existing_key).casefold() == wanted:
            return value
    return None


def parse_par_bytes(contents: bytes, file_name: str) -> CVFileData:
    """Parse an uploaded VersaStudio file entirely in memory."""

    text, encoding = decode_par_bytes(contents)
    metadata = extract_metadata(text)
    blocks = extract_segment_blocks(text)
    if not blocks:
        raise ParParseError("No <SegmentN> data blocks were found.")

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for order, (segment_number, body) in enumerate(blocks):
        frame, segment_warnings = _parse_segment(segment_number, body)
        warnings.extend(segment_warnings)
        if not frame.empty:
            frame["_segment_order"] = order
            frames.append(frame)
    if not frames:
        raise ParParseError("The file contains no usable E(V)/I(A) data rows.")

    raw_data = pd.concat(frames, ignore_index=True, sort=False)
    raw_data = raw_data.sort_values(["_segment_order", "_row_order"], kind="stable")
    raw_data = raw_data.reset_index(drop=True)
    file_hash = hashlib.sha256(contents).hexdigest()
    identity = f"{file_name}\0{len(contents)}\0{file_hash}".encode("utf-8")
    file_id = hashlib.sha256(identity).hexdigest()

    scan_rate = metadata_value(metadata, "Scan Rate (V/s)")
    if scan_rate is None:
        warnings.append("Scan Rate (V/s) metadata was not found.")
    else:
        try:
            metadata["Scan Rate (V/s)"] = float(scan_rate)
        except (TypeError, ValueError):
            metadata["Scan Rate (V/s)"] = None
            warnings.append("Scan Rate (V/s) metadata is not numeric.")

    return CVFileData(
        file_id=file_id,
        file_name=file_name,
        metadata=metadata,
        raw_data=raw_data,
        warnings=warnings,
        file_hash=file_hash,
        file_size=len(contents),
        encoding=encoding,
    )


def parse_par_stream(stream: io.BufferedIOBase, file_name: str) -> CVFileData:
    """Parse a binary stream, retaining no temporary copy on disk."""

    return parse_par_bytes(stream.read(), file_name)
