from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from core.exporters import analysis_zip, summary_csv
from core.regression import aggregate_replicates, apply_blank_correction, linear_regression


def test_linear_regression() -> None:
    result = linear_regression([1, 2, 3, 4], [2.1, 4.1, 6.1, 8.1])
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(0.1)
    assert result.r_squared == pytest.approx(1.0)


def test_through_origin_regression() -> None:
    result = linear_regression([1, 2, 3], [2, 4, 6], through_origin=True)
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == 0


def test_regression_requires_two_distinct_x_values() -> None:
    with pytest.raises(ValueError, match="distinct"):
        linear_regression([1, 1], [2, 3])


def test_missing_concentration_is_excluded_before_regression() -> None:
    data = pd.DataFrame({"concentration": [1.0, None, 3.0], "response": [2.0, 9.0, 6.0]})
    included = data.dropna(subset=["concentration"])
    result = linear_regression(included["concentration"], included["response"])
    assert result.n == 2
    assert 9.0 not in included["response"].tolist()


def test_replicate_mean_and_standard_deviation() -> None:
    data = pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": [2.0, 4.0, 8.0]})
    grouped = aggregate_replicates(data, mode="Mean ± standard deviation")
    first = grouped[grouped["x"] == 1.0].iloc[0]
    assert first["mean"] == pytest.approx(3.0)
    assert first["std"] == pytest.approx(2**0.5)


def test_zero_concentration_blank_is_average() -> None:
    data = pd.DataFrame(
        {
            "file_id": ["b1", "b2", "s"],
            "concentration_base_value": [0.0, 0.0, 1.0],
            "response": [1.0, 3.0, 8.0],
        }
    )
    corrected, blank = apply_blank_correction(data, "Use concentration 0 data")
    assert blank == pytest.approx(2.0)
    assert corrected.loc[2, "response"] == pytest.approx(6.0)


def test_summary_csv_is_utf8_sig() -> None:
    exported = summary_csv(pd.DataFrame({"file_name": ["\uc2e4\ud5d8.par"]}))
    assert exported.startswith(b"\xef\xbb\xbf")
    assert "file_hash" in exported.decode("utf-8-sig")


def test_integrated_zip_contains_required_files() -> None:
    payload = analysis_zip(None, None, None, None, None)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "summary.csv",
            "processed_cv_data.csv",
            "regression_results.csv",
            "cv_overlay.html",
            "calibration_plot.html",
        }
