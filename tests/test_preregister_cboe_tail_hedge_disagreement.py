from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_cboe_tail_hedge_disagreement as prereg


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
    assert selected["2022"]["events"] >= 30
    assert selected["2023_h2"]["events"] >= 20


def test_selected_tail_is_first_source_support_pass() -> None:
    payload = prereg.build_manifest()
    grid = payload["research_history_boundary"]["disclosure"][
        "upper_tail_support_grid"
    ]

    def passes(item: dict[str, dict[str, float]]) -> bool:
        return bool(
            item["stage1"]["events"] >= 150
            and item["2021"]["events"] >= 30
            and item["2022"]["events"] >= 30
            and item["2023"]["events"] >= 140
            and item["2023_h1"]["events"] >= 20
            and item["2023_h2"]["events"] >= 20
            and item["stage1"]["max_single_month_share"] <= 0.16
            and item["2023"]["max_single_month_share"] <= 0.16
        )

    passing = [float(tail) for tail, counts in grid.items() if passes(counts)]
    assert min(passing) == prereg.Policy.upper_tail_rank


def test_write_once_refuses_policy_change(tmp_path: Path) -> None:
    path = tmp_path / "prereg.json"
    payload = prereg.build_manifest()
    assert prereg.write_once(path, payload) == "created"
    assert prereg.write_once(path, payload) == "verified_existing"
    changed = json.loads(json.dumps(payload))
    changed["policy"]["upper_tail_rank"] = 0.25
    with pytest.raises(RuntimeError):
        prereg.write_once(path, changed)


def test_manifest_hash_detects_mutation() -> None:
    payload = prereg.build_manifest()
    payload["controls"]["direction_flip"] = "mutated"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.validate_manifest(payload, verify_sources=False)
