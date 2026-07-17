from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from training import preregister_cross_collateral_positioning_recoil as prereg


def test_manifest_is_deterministic_and_outcome_blind() -> None:
    first = prereg.build_manifest()
    second = prereg.build_manifest()
    assert first == second
    assert first["outcomes_opened"] is False
    assert first["causal_feature_contract"]["price_signal_columns"] == []
    assert first["selection_protocol"]["candidate_count_after_support"] == 2
    assert first["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
    prereg.validate_manifest(first)


def test_policy_freezes_causal_windows_grid_holds_and_delay() -> None:
    policy = asdict(prereg.Policy())
    assert policy["anchor_minute"] == 55
    assert policy["oi_change_bars"] == 72
    assert policy["taker_median_bars"] == 12
    assert policy["prior_rank_hourly_anchors"] == 168
    assert policy["rotation_quantiles"] == (0.80, 0.85, 0.90)
    assert policy["taker_rank_floor"] == 0.60
    assert policy["execution_delay_bars"] == 2
    assert policy["hold_bars"] == (48, 96)
    availability = prereg.build_manifest()["causal_feature_contract"]["availability"]
    assert "t+10m" in availability


def test_feature_contract_is_cross_collateral_and_price_blind() -> None:
    manifest = prereg.build_manifest()
    feature = manifest["causal_feature_contract"]
    assert feature["source_only_columns"] == [
        "um_sum_open_interest_value",
        "cm_sum_open_interest",
        "um_sum_taker_long_short_vol_ratio",
        "cm_sum_taker_long_short_vol_ratio",
        "source_complete",
    ]
    assert "current t is excluded" in feature["strict_prior_ranks"]
    assert "side[t]=-sign(T[t])" in feature["action"]
    assert "OHLC_or_price_return" in manifest["novelty_boundary"]["excluded_inputs"]
    assert (
        "existing_alpha_state_or_portfolio_PnL"
        in manifest["novelty_boundary"]["excluded_inputs"]
    )


def test_source_only_density_disclosure_and_selection_are_frozen() -> None:
    manifest = prereg.build_manifest()
    disclosure = manifest["research_history_boundary"]["density_preflight_disclosure"]
    assert disclosure["episode_counts_by_q"]["0.85"] == {
        "2021_partial": 35,
        "2022": 78,
        "2023": 49,
    }
    support = manifest["support_calibration"]
    assert support["vary_only"] == "rotation quantile Q in [0.80, 0.85, 0.90]"
    assert support["minimum_train_episodes"] == 100
    assert support["minimum_each_2023_half"] == 10
    assert "never select an outcome-driven fallback" in support["selection_rule"]


def test_controls_and_complete_battery_are_frozen() -> None:
    manifest = prereg.build_manifest()
    controls = manifest["falsification_controls"]
    assert set(controls) == {
        "oi_only",
        "taker_only",
        "um_only",
        "cm_only",
        "direction_flip",
        "entry_shift_plus_1h",
        "deterministic_random_side",
    }
    comparison = manifest["selection_protocol"]["control_comparison_contract"]
    assert set(comparison["mechanism_controls"]) == {
        "oi_only",
        "taker_only",
        "um_only",
        "cm_only",
    }
    assert "complete" in comparison["all_controls"]
    assert (
        manifest["selection_protocol"]["statistical_test_contract"]["draws"] == 20_000
    )


def test_orthogonality_comparator_artifacts_are_frozen() -> None:
    comparators = prereg.build_manifest()["orthogonality_after_standalone_pass"][
        "comparator_universe"
    ]
    assert len(comparators) == 4
    for item in comparators.values():
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def test_validate_manifest_rejects_outcomes_policy_drift_and_hash_drift() -> None:
    opened = prereg.build_manifest()
    opened["outcomes_opened"] = True
    opened["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in opened.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="outcomes opened"):
        prereg.validate_manifest(opened, verify_sources=False)

    policy_drift = prereg.build_manifest()
    policy_drift["policy"]["hold_bars"] = [24, 48]
    policy_drift["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in policy_drift.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy differs"):
        prereg.validate_manifest(policy_drift, verify_sources=False)

    hash_drift = prereg.build_manifest()
    hash_drift["support_calibration"]["minimum_train_episodes"] = 1
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        prereg.validate_manifest(hash_drift, verify_sources=False)


def test_frozen_source_hashes_match() -> None:
    source = prereg.build_manifest()["source_contract"]
    for key, hash_key in (
        ("positioning", "positioning_sha256"),
        ("positioning_manifest", "positioning_manifest_sha256"),
        ("market", "market_sha256"),
        ("market_manifest", "market_manifest_sha256"),
        ("funding", "funding_sha256"),
        ("funding_manifest", "funding_manifest_sha256"),
    ):
        assert (
            hashlib.sha256(Path(source[key]).read_bytes()).hexdigest()
            == source[hash_key]
        )


def test_written_preregistration_artifact_replays() -> None:
    payload = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    assert payload == prereg.build_manifest()
    prereg.validate_manifest(payload)
