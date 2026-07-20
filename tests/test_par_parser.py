from __future__ import annotations

import pytest

from core.par_parser import (
    ParParseError,
    decode_par_bytes,
    extract_metadata,
    extract_segment_blocks,
    parse_par_bytes,
)


def test_scan_rate_extraction(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "2.5mM.par")
    assert parsed.metadata["Scan Rate (V/s)"] == pytest.approx(0.05)


def test_cycle_count_extraction(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    assert parsed.metadata["Cycles"] == 2


def test_segment_blocks_extracted(par_bytes: bytes) -> None:
    text, _ = decode_par_bytes(par_bytes)
    assert [number for number, _ in extract_segment_blocks(text)] == [1, 2, 3, 4]


def test_potential_and_current_parsed(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    assert {"potential_V", "current_A"}.issubset(parsed.raw_data.columns)
    assert parsed.raw_data.iloc[2]["potential_V"] == pytest.approx(0.2)


def test_scientific_notation_parsed(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    assert parsed.raw_data.iloc[2]["current_A"] == pytest.approx(5e-6)


def test_corrupt_row_is_skipped_with_warning(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    assert len(parsed.raw_data) == 16
    assert any("skipped 1 row" in warning for warning in parsed.warnings)


def test_multiple_segments_are_merged(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "sample.par")
    assert parsed.raw_data["segment_number"].nunique() == 4


def test_file_name_does_not_create_concentration(par_bytes: bytes) -> None:
    parsed = parse_par_bytes(par_bytes, "2.5mM.par")
    assert "Concentration" not in parsed.metadata
    assert not hasattr(parsed, "concentration")


def test_hash_identity_uses_name_size_and_content(par_bytes: bytes) -> None:
    first = parse_par_bytes(par_bytes, "a.par")
    repeat = parse_par_bytes(par_bytes, "a.par")
    renamed = parse_par_bytes(par_bytes, "b.par")
    assert first.file_id == repeat.file_id
    assert first.file_id != renamed.file_id
    assert len(first.file_hash) == 64


def test_cp949_decoding() -> None:
    raw = "\uc2e4\ud5d8=CV\n".encode("cp949")
    text, encoding = decode_par_bytes(raw)
    assert "\uc2e4\ud5d8" in text
    assert encoding == "cp949"


def test_file_without_segments_fails() -> None:
    with pytest.raises(ParParseError, match="Segment"):
        parse_par_bytes(b"Cycles=1\nScan Rate (V/s)=0.1\n", "bad.par")


def test_global_metadata_excludes_segment_settings(par_bytes: bytes) -> None:
    text, _ = decode_par_bytes(par_bytes)
    metadata = extract_metadata(text)
    assert "Type" not in metadata


def test_numeric_definition_sentinel_is_not_treated_as_data_column() -> None:
    raw = b"""Cycles=1
Scan Rate (V/s)=0.05
Segments=1
<Segment1>
Definition=Segment #,Point #,E(V),I(A),Elapsed Time(s),E Applied(V),0
0,0,0.0,1E-6,0.0,0.0
0,1,0.1,2E-6,1.0,0.1
</Segment1>
"""
    parsed = parse_par_bytes(raw, "real-format.par")
    assert len(parsed.raw_data) == 2
    assert "0" not in parsed.raw_data.columns
    assert parsed.raw_data["segment_number"].tolist() == [0, 0]
