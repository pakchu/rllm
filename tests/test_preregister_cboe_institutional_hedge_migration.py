from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_cboe_institutional_hedge_migration as prereg


def test_manifest_is_singleton_outcome_blind_and_source_only() -> None:
    payload = prereg.build_manifest()
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["policy"]["direction"] == "SHORT_ONLY"
    assert payload["causal_feature_contract"][
        "price_or_derivative_feature_columns_loaded"
    ] == []
    disclosure = payload["research_history_boundary"]["disclosure"]
    assert disclosure["outcomes_joined"] is False
    assert disclosure["market_rows_loaded"] == 0
    selected = disclosure["selected_clocks"]["primary"]
    assert selected["stage1"]["events"] >= 150
    assert selected["2021"]["events"] >= 70
    assert selected["2022"]["events"] >= 70
    assert selected["2023"]["events"] >= 60
    assert selected["2023_h1"]["events"] >= 25
    assert selected["2023_h2"]["events"] >= 25


def test_selected_threshold_is_highest_source_support_pass() -> None:
    payload = prereg.build_manifest()
    grid = payload["research_history_boundary"]["disclosure"][
        "composite_threshold_support_grid"
    ]

    def passes(item: dict[str, dict[str, float]]) -> bool:
        return bool(
            item["stage1"]["events"] >= 150
            and item["2021"]["events"] >= 70
            and item["2022"]["events"] >= 70
            and item["2023"]["events"] >= 60
            and item["2023_h1"]["events"] >= 25
            and item["2023_h2"]["events"] >= 25
            and item["stage1"]["max_single_month_share"] <= 0.15
            and item["2023"]["max_single_month_share"] <= 0.15
        )

    passing = [float(threshold) for threshold, counts in grid.items() if passes(counts)]
    assert max(passing) == prereg.Policy.composite_threshold


def test_write_once_refuses_policy_change(tmp_path: Path) -> None:
    path = tmp_path / "prereg.json"
    payload = prereg.build_manifest()
    assert prereg.write_once(path, payload) == "created"
    assert prereg.write_once(path, payload) == "verified_existing"
    changed = json.loads(json.dumps(payload))
    changed["policy"]["composite_threshold"] = 0.58
    with pytest.raises(RuntimeError):
        prereg.write_once(path, changed)


def test_manifest_hash_detects_mutation() -> None:
    payload = prereg.build_manifest()
    payload["controls"]["direction_flip"] = "mutated"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.validate_manifest(payload, verify_sources=False)


def test_frozen_preregistration_and_clock_identities() -> None:
    path = Path(prereg.DEFAULT_OUTPUT)
    payload = json.loads(path.read_text())
    prereg.validate_manifest(payload)
    assert prereg.sha256_file(path) == (
        "0709c7aff57dc1e1e7079979ec44ceb0e154c47898ea593f2bfe50d1ab4052d5"
    )
    assert payload["manifest_hash"] == (
        "3dd49c08c685a191f686f42cf6a27af30d057ea92529cc4008db54a4980582fe"
    )
    assert prereg.sha256_file(prereg.clock.DEFAULT_OUTPUT) == (
        "188196f1ea8d6ecd741306419e540b9ec9c11800d9b96d3d2ad591cc3fc94cf0"
    )
