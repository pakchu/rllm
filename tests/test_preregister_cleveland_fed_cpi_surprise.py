from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_cleveland_fed_cpi_surprise as prereg


def test_manifest_is_singleton_outcome_blind_and_source_only() -> None:
    payload = prereg.build_manifest()
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_outcome_windows"] == []
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert (
        payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"]
        == []
    )
    disclosure = payload["research_history_boundary"]["disclosure"]
    assert disclosure["outcomes_joined"] is False
    assert disclosure["outcome_sources_opened"] == []
    assert disclosure["market_rows_loaded"] == 0
    assert disclosure["funding_rows_loaded"] == 0
    primary = disclosure["selected_clocks"]["primary"]
    assert primary["stage1"]["events"] == 26
    assert primary["stage1"]["longs"] == 10
    assert primary["stage1"]["shorts"] == 16
    assert primary["2023"]["events"] == 8


def test_selected_threshold_is_highest_source_support_pass() -> None:
    payload = prereg.build_manifest()
    grid = payload["research_history_boundary"]["disclosure"]["threshold_support_grid"]

    def passes(item: dict[str, dict[str, float]]) -> bool:
        return bool(
            item["stage1"]["events"] >= 24
            and item["2020"]["events"] >= 8
            and item["2021"]["events"] >= 8
            and item["2022"]["events"] >= 8
            and item["2023"]["events"] >= 8
            and item["2023_h1"]["events"] >= 4
            and item["2023_h2"]["events"] >= 4
            and item["stage1"]["max_single_month_share"] <= 0.125
            and item["2023"]["max_single_month_share"] <= 0.125
        )

    passing = [float(threshold) for threshold, counts in grid.items() if passes(counts)]
    assert max(passing) == prereg.Policy.threshold_pct


def test_write_once_refuses_policy_change(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"
    payload = prereg.canonical_json(prereg.build_manifest())
    assert prereg.write_once(output, payload) == "created"
    assert prereg.write_once(output, payload) == "verified_existing"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        prereg.write_once(output, payload + b"\n")


def test_manifest_hash_detects_mutation() -> None:
    payload = prereg.build_manifest()
    payload["controls"]["falsification"]["direction_flip"] = "mutated"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.validate_manifest(payload, verify_sources=False)


def test_frozen_preregistration_and_clock_identities() -> None:
    path = Path(prereg.DEFAULT_OUTPUT)
    payload = json.loads(path.read_text())
    prereg.validate_manifest(payload)
    assert prereg.sha256_file(path) == (
        "9c252a988885c7fa1975b6f7190af4efeab50ee8541a67c0bb8f8882a3fa3e0d"
    )
    assert payload["manifest_hash"] == (
        "61604984515942428977d24afa39299a5083e8e2b36fef3ac7cf95b4eddf6b60"
    )
    assert prereg.sha256_file(prereg.DEFAULT_CLOCK) == (
        "cff8d0f8d7810400bc78f833cc91996a7b2cd0e9d5903fe0ef154f0e38a71739"
    )
