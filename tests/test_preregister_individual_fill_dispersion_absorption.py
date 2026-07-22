from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from training import preregister_individual_fill_dispersion_absorption as ifda


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = ifda.build_manifest()
    ifda.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["policy"]["policy_id"] == "IFDA-72"
    assert payload["feature_contract"]["candidate_count"] == 1


def test_policy_freezes_individual_fill_formula_and_fade_direction() -> None:
    payload = ifda.build_manifest()
    feature = payload["feature_contract"]
    policy = payload["policy"]
    assert feature["side_hhi"].startswith("sum(q_i^2)")
    assert feature["score"] == (
        "flow_coherence*dominant_equalization*equalization_gap"
    )
    assert feature["direction"].startswith("-dominant_side")
    assert feature["price_path_fields_used"] is False
    assert feature["aggregate_trade_event_fields_used_by_primary"] is False
    assert policy["score_quantile"] == 0.995
    assert policy["hold_bars"] == 72
    assert policy["execution_delay_bars"] == 2


def test_source_contract_is_checksum_bound_streamed_and_disk_guarded() -> None:
    payload = ifda.build_manifest()
    source = payload["source_contract"]
    assert source["checksum_suffix"] == ".CHECKSUM"
    assert source["raw_archives_persisted"] is False
    assert source["range"] == ["2020-01-01", "2024-01-01"]
    assert "exact +1 continuity" in source["trade_id_contract"]
    assert "300 GiB" in source["disk_guard"]
    assert payload["policy"]["disk_used_abort_gib"] == 300


def test_support_and_controls_require_granularity_increment() -> None:
    payload = ifda.build_manifest()
    controls = payload["falsification_controls"]
    economics = payload["later_economic_protocol"]
    assert "aggtrade_equalization" in controls
    assert "flow_only_fade" in controls
    assert "remove_cross_side_asymmetry" in controls
    assert "all_fill_equalization" in controls
    increment = economics["granularity_increment_gate"]
    assert increment["required_stages"] == ["train", "selection", "test", "eval"]
    assert increment["ratio_margin_min"] == 0.50
    assert increment["all_stages_must_pass_independently"] is True
    assert economics["ratio_definition"]["formula"] == "cagr_pct/strict_mdd_pct"


def test_control_clock_bundle_and_hash_contract_are_mandatory() -> None:
    contract = ifda.build_manifest()["support_artifact_contract"]
    assert contract["required_own_clock_controls"] == list(ifda.OWN_CLOCK_CONTROL_IDS)
    assert contract["control_clock_columns"][0] == "control_id"
    assert contract["missing_control_effect"] == "REJECT_NO_REPAIR"
    assert "control_clock_bundle_sha256" in contract["required_result_hash_fields"]


def test_oos_control_contract_mutation_is_rejected() -> None:
    payload = ifda.build_manifest()
    payload["later_economic_protocol"]["granularity_increment_gate"][
        "required_stages"
    ] = ["train", "selection"]
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    payload["manifest_hash"] = ifda.canonical_hash(core)
    with pytest.raises(RuntimeError, match="frozen contract differs"):
        ifda.validate_manifest(payload)


def test_novelty_contract_binds_prior_bundle_and_recent_clocks() -> None:
    payload = ifda.build_manifest()
    novelty = payload["novelty_gates"]
    assert novelty["prior_bundle"]["sha256"] == ifda.PRIOR_COMPARATOR_BUNDLE_SHA256
    families = [item["family"] for item in novelty["explicit_clocks"]]
    assert families[-2:] == ["VTMS", "QLCD"]
    assert novelty["exact_entry_jaccard_max"] == 0.05
    assert novelty["tolerant_one_to_one_jaccard_max"] == 0.15


def test_manifest_builds_do_not_share_mutable_comparator_state() -> None:
    first = ifda.build_manifest()
    original_family = first["novelty_gates"]["explicit_clocks"][0]["family"]
    first["novelty_gates"]["explicit_clocks"][0]["family"] = "MUTATED"
    second = ifda.build_manifest()
    assert second["novelty_gates"]["explicit_clocks"][0]["family"] == original_family


def test_recomputed_hash_cannot_change_nonpolicy_contract() -> None:
    payload = ifda.build_manifest()
    payload["source_contract"]["range"] = ["2021-01-01", "2024-01-01"]
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    payload["manifest_hash"] = ifda.canonical_hash(core)
    with pytest.raises(RuntimeError, match="frozen contract differs"):
        ifda.validate_manifest(payload)


def test_referenced_artifact_hashes_match_repository_bytes() -> None:
    payload = ifda.build_manifest()
    decision = payload["source_axis_decision"]
    assert hashlib.sha256(Path(decision["path"]).read_bytes()).hexdigest() == decision["sha256"]
    bundle = payload["novelty_gates"]["prior_bundle"]
    assert hashlib.sha256(Path(bundle["path"]).read_bytes()).hexdigest() == bundle["sha256"]
    for item in payload["novelty_gates"]["explicit_clocks"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_live_parity_does_not_assume_raw_futures_websocket() -> None:
    gate = ifda.build_manifest()["live_parity_gate"]
    assert gate["raw_futures_websocket_assumed"] is False
    assert gate["recent_endpoint"] == "/fapi/v1/trades"
    assert gate["catchup_endpoint"] == "/fapi/v1/historicalTrades"
    assert "exclude from IFDA feature math" in gate["schema_normalization"]["isRPITrade"]
    assert gate["request_contract"]["historical_lookback_max"] == "one month"
    assert gate["request_contract"]["unresolved_gap_wall_clock_minutes_max"] == 15
    assert gate["gap_behavior"].startswith("suppress new orders")
    assert gate["failure_effect"].startswith("research-only")


def test_live_schema_mutation_is_rejected_even_with_recomputed_hash() -> None:
    payload = ifda.build_manifest()
    payload["live_parity_gate"]["schema_normalization"]["isRPITrade"] = "use as alpha"
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    payload["manifest_hash"] = ifda.canonical_hash(core)
    with pytest.raises(RuntimeError, match="frozen contract differs"):
        ifda.validate_manifest(payload)


def test_manifest_hash_detects_mutation() -> None:
    payload = ifda.build_manifest()
    payload["policy"]["hold_bars"] = 144
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ifda.validate_manifest(payload)


def test_write_once_refuses_changed_policy(tmp_path: Path) -> None:
    path = tmp_path / "ifda.json"
    assert ifda.write_manifest_once(path, ifda.build_manifest()) == "created"
    assert ifda.write_manifest_once(path, ifda.build_manifest()) == "verified_existing"
    changed = json.loads(path.read_text())
    changed["policy"]["hold_bars"] = 144
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = ifda.canonical_hash(core)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="frozen policy differs"):
        ifda.write_manifest_once(path, ifda.build_manifest())


def test_repository_artifact_matches_code() -> None:
    artifact = json.loads(Path(ifda.DEFAULT_OUTPUT).read_text())
    ifda.validate_manifest(artifact)
    assert artifact["manifest_hash"] == ifda.build_manifest()["manifest_hash"]
