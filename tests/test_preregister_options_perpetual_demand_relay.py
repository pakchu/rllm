from __future__ import annotations

from copy import deepcopy

import pytest

from training import preregister_options_perpetual_demand_relay as prereg


def test_manifest_is_deterministic_singleton_and_outcome_blind() -> None:
    first = prereg.build_manifest()
    second = prereg.build_manifest()
    assert first == second
    prereg.validate_manifest(first, verify_sources=False)
    assert first["outcomes_opened"] is False
    assert first["research_history_boundary"]["candidate_count"] == 1
    assert first["research_history_boundary"]["threshold_grid"] is False
    assert first["research_history_boundary"]["direction_search"] is False
    assert first["research_history_boundary"]["hold_search"] is False


def test_clock_excludes_btc_price_and_uses_premium_for_direction() -> None:
    report = prereg.build_manifest()
    forbidden = report["source_contract"]["clock_forbidden_fields"]
    assert "BTCUSDT_price" in forbidden
    assert "BTCUSDT_return" in forbidden
    assert report["mechanism"]["action"] == (
        "side equals the sign of the completed one-hour premium move"
    )
    assert "excludes BTC price and return" in report["mechanism"][
        "why_distinct_from_dvol_price_follow"
    ]


def test_policy_freezes_strict_prior_features_and_execution() -> None:
    report = prereg.build_manifest()
    policy = report["policy"]
    assert policy["prior_window_hours"] == 720
    assert policy["prior_min_periods_hours"] == 672
    assert policy["bvol_dvol_ratio_low_quantile"] == 0.20
    assert policy["premium_move_abs_quantile"] == 0.80
    assert policy["premium_efficiency_quantile"] == 0.70
    assert policy["entry_delay_minutes"] == 5
    assert policy["hold_hours"] == 24
    assert "excludes the current hour" in report["causal_feature_contract"][
        "strict_prior_thresholds"
    ]


def test_sequential_oos_contract_stops_on_first_failure() -> None:
    report = prereg.build_manifest()
    assert report["splits"] == {
        "train": ["2023-07-01", "2024-01-01"],
        "test": ["2024-01-01", "2025-01-01"],
        "eval": ["2025-01-01", "2026-01-01"],
        "final": ["2026-01-01", "2026-07-01"],
    }
    assert report["outcome_gate"]["sequential_opening"] == (
        "train_then_test_then_eval_then_final_stop_on_first_failure"
    )
    assert report["outcome_gate"]["cagr_to_strict_mdd_min"] == 3.0
    assert report["outcome_gate"]["strict_mdd_max_pct"] == 15.0


def test_manifest_rejects_mutation() -> None:
    report = prereg.build_manifest()
    changed = deepcopy(report)
    changed["policy"]["hold_hours"] = 48
    with pytest.raises(ValueError, match="policy changed"):
        prereg.validate_manifest(changed, verify_sources=False)


def test_manifest_rejects_rehashed_threshold_grid() -> None:
    report = prereg.build_manifest()
    changed = deepcopy(report)
    changed["research_history_boundary"]["threshold_grid"] = True
    core = {key: value for key, value in changed.items() if key != "manifest_hash"}
    changed["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(ValueError, match="threshold grid"):
        prereg.validate_manifest(changed, verify_sources=False)
