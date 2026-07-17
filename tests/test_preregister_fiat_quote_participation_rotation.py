from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from training import preregister_fiat_quote_participation_rotation as prereg


def test_manifest_is_deterministic_singleton_and_outcome_blind() -> None:
    first = prereg.build_manifest()
    second = prereg.build_manifest()
    assert first == second
    assert first["outcomes_opened"] is False
    assert first["selection_protocol"]["candidate_count"] == 1
    assert first["causal_feature_contract"]["price_signal_columns"] == []
    boundary = first["research_history_boundary"]
    assert boundary["full_2021_2023_source_support_seen_before_freeze"] is True
    assert "outcome-sealed" in boundary["consequence"]
    assert first["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
    prereg.validate_manifest(first)


def test_policy_freezes_daily_availability_breadth_hold_and_only_grid() -> None:
    policy = asdict(prereg.Policy())
    assert policy["baseline_days"] == 180
    assert policy["baseline_min_periods"] == 180
    assert policy["fiat_quote_symbols"] == ("BTCEUR", "BTCTRY", "BTCBRL")
    assert policy["participation_quantiles"] == (0.50, 0.55, 0.60, 0.65, 0.70)
    assert policy["minimum_breadth"] == 2
    assert policy["execution_delay_bars"] == 1
    assert policy["hold_bars"] == 864
    manifest = prereg.build_manifest()
    assert manifest["support_calibration"]["vary_only"].startswith(
        "participation quantile"
    )
    assert "cannot select a fallback" in manifest["support_calibration"][
        "selection_rule"
    ]
    assert manifest["selection_protocol"]["stage2_support_cannot_reselect_q"] is True
    assert "d+1 00:05 UTC" in manifest["causal_feature_contract"]["availability"]


def test_feature_contract_has_no_price_or_fx_and_requires_strict_prior_rank() -> None:
    manifest = prereg.build_manifest()
    feature = manifest["causal_feature_contract"]
    assert feature["source_only_columns"] == [
        "base_volume_btc",
        "trade_count",
        "taker_buy_base_btc",
        "taker_sell_base_btc",
    ]
    assert "d-180..d-1" in feature["lagged_ranks"]
    assert "current d is excluded" in feature["lagged_ranks"]
    excluded = manifest["novelty_boundary"]["excluded_inputs"]
    assert "OHLC_or_price_return" in excluded
    assert "kimchi_BTCKRW_USDKRW_or_DXY" in excluded
    assert feature["participation_score"] == "P_r[d]=(R_V_r[d]+R_N_r[d])/2"
    assert "median" in feature["relative_taker_pressure"]


def test_support_and_performance_counts_are_identical_and_splits_are_causal() -> None:
    manifest = prereg.build_manifest()
    support = manifest["support_calibration"]
    gates = manifest["selection_protocol"]["gates"]
    assert support["minimum_nonoverlap_train"] == gates["train_trades_min"] == 40
    assert (
        support["minimum_2021_after_warmup"]
        == gates["2021_after_warmup_trades_min"]
        == 20
    )
    assert support["minimum_2022"] == gates["2022_trades_min"] == 18
    assert support["minimum_2023"] == gates["2023_trades_min"] == 20
    assert (
        support["minimum_each_2023_half"]
        == gates["2023_h1_and_h2_trades_min"]
        == 8
    )
    reservation = manifest["execution_contract"]["clock_reservation"]
    assert "history may precede a split start" in reservation
    assert "signal, entry, and exit" in reservation


def test_controls_statistical_test_and_comparison_are_fully_frozen() -> None:
    manifest = prereg.build_manifest()
    support_controls = set(
        manifest["support_calibration"]["maximum_signal_jaccard"]
    )
    frozen_controls = set(manifest["falsification_controls"])
    assert support_controls <= frozen_controls
    comparison = manifest["selection_protocol"]["control_comparison_contract"]
    assert set(comparison["mechanism_controls"]) <= frozen_controls
    assert "equality rejects" in comparison["stage1"]
    assert "min(primary train ratio" in comparison["stage2"]
    statistical = manifest["selection_protocol"]["statistical_test_contract"]
    assert statistical["draws"] == 20_000
    assert statistical["seed"] == 20_260_717


def test_orthogonality_comparator_artifacts_are_frozen() -> None:
    comparators = prereg.build_manifest()["orthogonality_after_standalone_pass"][
        "comparator_universe"
    ]
    assert len(comparators) == 4
    for item in comparators.values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]


def test_validate_manifest_rejects_drift_and_opened_outcomes() -> None:
    opened = prereg.build_manifest()
    opened["outcomes_opened"] = True
    opened["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in opened.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="outcomes opened"):
        prereg.validate_manifest(opened, verify_sources=False)

    drifted = prereg.build_manifest()
    drifted["policy"]["hold_bars"] = 288
    drifted["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in drifted.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy differs"):
        prereg.validate_manifest(drifted, verify_sources=False)


def test_frozen_source_hashes_match() -> None:
    source = prereg.build_manifest()["source_contract"]
    for key, hash_key in (
        ("flow", "flow_sha256"),
        ("flow_manifest", "flow_manifest_sha256"),
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
