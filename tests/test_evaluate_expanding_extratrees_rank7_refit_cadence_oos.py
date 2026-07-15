from __future__ import annotations

import json

import pytest

from training.evaluate_expanding_extratrees_rank7_refit_cadence_oos import (
    EXPECTED_CADENCE_MANIFEST_HASH,
    divergent_trade_performance,
    schedule_overlap,
    validate_cadence_manifest,
)


def test_frozen_cadence_manifest_selects_annual() -> None:
    manifest, result = validate_cadence_manifest()
    assert manifest["manifest_hash"] == EXPECTED_CADENCE_MANIFEST_HASH
    assert manifest["selected_cadence"] == "annual"
    assert result["selected_cadence"] == "annual"
    assert result["cadences"]["annual"]["selection_pass"] is True
    assert result["cadences"]["monthly"]["selection_pass"] is False


def test_tampered_cadence_manifest_is_rejected(tmp_path) -> None:
    manifest, _ = validate_cadence_manifest()
    manifest["selected_cadence"] = "monthly"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        validate_cadence_manifest(str(path))


def test_schedule_overlap_reports_shared_and_unique_entries() -> None:
    overlap = schedule_overlap(
        {"x": {1: 1.0, 2: 2.0, 3: 3.0}},
        {"x": {2: 2.0, 3: 3.0, 4: 4.0}},
    )["x"]
    assert overlap == {
        "annual_entries": 3,
        "monthly_entries": 3,
        "shared_entries": 2,
        "union_entries": 4,
        "jaccard": 0.5,
    }


def test_divergent_trade_performance_separates_schedule_edges() -> None:
    result = divergent_trade_performance(
        {"x": {1: 10.0, 2: 20.0}},
        {"x": {2: 20.0, 3: -5.0}},
    )["x"]
    assert result["shared"] == {"trades": 1, "mean_net_bps": 20.0, "win_rate": 1.0}
    assert result["annual_only"] == {"trades": 1, "mean_net_bps": 10.0, "win_rate": 1.0}
    assert result["monthly_only"] == {"trades": 1, "mean_net_bps": -5.0, "win_rate": 0.0}
