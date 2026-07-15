from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cross_collateral_near_pressure_oos as evaluator


def test_frozen_selection_manifest_replays() -> None:
    manifest, result = evaluator.validate_selection_manifest(evaluator.SELECTION_MANIFEST)
    assert manifest["selected_spec"] == evaluator.EXPECTED_SELECTED
    assert manifest["future_outcomes_opened"] is False
    assert result["grid_cells"] == 104


def test_tampered_selection_manifest_is_rejected(tmp_path) -> None:
    payload = json.loads(open(evaluator.SELECTION_MANIFEST).read())
    payload["selected_spec"]["hold_bars"] = 12
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="file hash"):
        evaluator.validate_selection_manifest(str(path))


def test_align_score_preserves_book_clock_and_leaves_prehistory_empty() -> None:
    book = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3, freq="5min")})
    score = pd.Series([1.0, 2.0, 3.0])
    full = pd.Series(pd.date_range("2022-12-31 23:55", periods=4, freq="5min"))
    aligned = evaluator.align_score(book, score, full)
    assert np.isnan(aligned.iloc[0])
    assert aligned.iloc[1:].tolist() == [1.0, 2.0, 3.0]


def test_future_manifest_rejects_outcome_bearing_panel(monkeypatch, tmp_path) -> None:
    data_path = tmp_path / "panel.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "protocol": {
            "outcomes_opened": False,
            "price_or_return_loaded": False,
            "raw_archives_retained": False,
            "checksums_verified": True,
            "start_inclusive": "2024-01-01",
            "end_exclusive": evaluator.FULL_CUTOFF,
        },
        "builder_sha256": evaluator.EXPECTED_BUILDER_SHA256,
        "dependency_sha256": evaluator.EXPECTED_BUILDER_DEPENDENCIES,
        "file": {
            "path": str(data_path),
            "sha256": "panel-hash",
            "rows": 1,
            "source_complete_rows": 1,
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "um_snapshot_count": [10],
            "um_first_offset_seconds": [0.0],
            "um_last_offset_seconds": [270.0],
            "um_near_pressure": [1.0],
            "cm_snapshot_count": [10],
            "cm_first_offset_seconds": [0.0],
            "cm_last_offset_seconds": [270.0],
            "cm_near_pressure": [1.0],
            "source_complete": [True],
            "close": [100.0],
        }
    )
    monkeypatch.setattr(
        evaluator,
        "sha256",
        lambda path: (
            evaluator.EXPECTED_FUTURE_BOOK_MANIFEST_SHA256
            if str(path) == str(manifest_path)
            else "panel-hash"
        ),
    )
    monkeypatch.setattr(evaluator, "resolve_existing", lambda path: data_path)
    monkeypatch.setattr(evaluator.pd, "read_csv", lambda *args, **kwargs: frame)
    with pytest.raises(RuntimeError, match="outcome-bearing column"):
        evaluator.validate_future_book_manifest(str(manifest_path))


def test_undefined_pnl_correlation_fails_closed() -> None:
    candidate = pd.Series([0.0] * 20)
    baseline = pd.Series([0.01, -0.01] * 10)
    result = evaluator.correlation_diagnostics(candidate, baseline)
    assert result["defined"] is False
    assert result["pearson"] is None


def test_execution_economics_drift_is_rejected() -> None:
    base = evaluator.ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv=evaluator.SelectionConfig().funding_csv,
        output="",
        manifest_output="",
    )
    with pytest.raises(RuntimeError, match="leverage"):
        evaluator.assert_execution_parity(base, replace(base, leverage=1.0))
