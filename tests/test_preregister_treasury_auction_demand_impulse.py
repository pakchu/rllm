from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_treasury_auction_demand_impulse as prereg


def test_manifest_is_deterministic_and_outcome_blind() -> None:
    first = prereg.build_manifest()
    second = prereg.build_manifest()
    assert first == second
    assert first["outcomes_opened"] is False
    assert first["causal_feature_contract"]["price_signal_columns"] == []
    assert first["selection_protocol"]["candidate_count"] == 1
    prereg.validate_manifest(first)


def test_source_only_counts_are_frozen_and_balanced_enough() -> None:
    counts = prereg.build_manifest()["support_gates"]["expected_counts"]
    assert counts["train"]["events"] == 28
    assert counts["train"]["side_counts"] == {"LONG": 18, "SHORT": 10}
    assert counts["2023"]["events"] == 23
    assert counts["2023"]["side_counts"] == {"LONG": 9, "SHORT": 14}
    assert counts["2023_h1"]["events"] == 12
    assert counts["2023_h2"]["events"] == 11


def test_feature_direction_and_no_bridge_are_frozen() -> None:
    manifest = prereg.build_manifest()
    feature = manifest["causal_feature_contract"]
    assert "exactly 12 prior valid same-tenor" in feature["strict_prior_ranks"]
    assert "LONG iff both ranks >=0.75" in feature["setup"]
    assert "never bridge" in manifest["source_contract"]["source_incomplete_policy"]
    assert "BTC_OHLC_or_return" in manifest["novelty_boundary"]["excluded_inputs"]


def test_controls_and_strict_stage_gate_are_frozen() -> None:
    manifest = prereg.build_manifest()
    assert set(manifest["falsification_controls"]) == {
        "bid_to_cover_only",
        "indirect_only",
        "direction_flip",
        "one_auction_delay",
        "deterministic_random_side",
    }
    gates = manifest["selection_protocol"]["gates"]
    assert gates["cagr_to_strict_mdd_min"] == 3.0
    assert gates["strict_mdd_pct_max"] == 15.0
    assert gates["weekly_cluster_signflip_p_max"] == 0.10


def test_validate_rejects_policy_outcome_and_hash_drift() -> None:
    opened = prereg.build_manifest()
    opened["outcomes_opened"] = True
    opened["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in opened.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="outcomes opened"):
        prereg.validate_manifest(opened, verify_sources=False)

    policy = prereg.build_manifest()
    policy["policy"]["hold_hours"] = 48
    policy["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in policy.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy differs"):
        prereg.validate_manifest(policy, verify_sources=False)

    hash_drift = prereg.build_manifest()
    hash_drift["support_gates"]["minimum_train_events"] = 1
    with pytest.raises(ValueError, match="hash mismatch"):
        prereg.validate_manifest(hash_drift, verify_sources=False)


def test_sources_and_written_artifact_replay() -> None:
    source = prereg.build_manifest()["source_contract"]
    for path_key, hash_key in (
        ("auction_panel", "auction_panel_sha256"),
        ("auction_manifest", "auction_manifest_sha256"),
        ("market", "market_sha256"),
        ("market_manifest", "market_manifest_sha256"),
        ("funding", "funding_sha256"),
        ("funding_manifest", "funding_manifest_sha256"),
    ):
        assert hashlib.sha256(Path(source[path_key]).read_bytes()).hexdigest() == source[hash_key]
    payload = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    assert payload == prereg.build_manifest()
    prereg.validate_manifest(payload)
