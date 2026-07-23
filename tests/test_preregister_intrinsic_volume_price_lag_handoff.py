from __future__ import annotations

import json

import pytest

from training import preregister_intrinsic_volume_price_lag_handoff as p


def test_manifest_is_incidence_comparator_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "IVPLH-72"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["predecessor_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["research_history_boundary"]["source_seen_successor"] is True


def test_policy_freezes_causal_latency_handoff_and_no_magnitude_gate() -> None:
    payload = p.build_manifest()
    policy = payload["policy"]
    feature = payload["causal_feature_contract"]
    execution = payload["execution_contract"]
    assert policy["intrinsic_volume_fraction"] == 0.50
    assert policy["event_reference_anchors"] == 180
    assert policy["event_reference_min_anchors"] == 90
    assert policy["decision_delay_bars"] == 1
    assert policy["entry_delay_bars"] == 2
    assert policy["hold_bars"] == 72
    assert feature["flow_magnitude_threshold"] is None
    assert feature["prior_run_minimum"] is None
    assert "negative previous flow side" in feature["handoff"]
    assert "<= 0" in feature["price_lag"]
    assert feature["future_bar_used_by_signal"] is False
    assert execution["entry_time"].endswith("+ 10m")


def test_lineage_is_hash_bound_and_requires_exact_66_row_identity() -> None:
    lineage = p.build_manifest()["predecessor_lineage"]
    assert lineage["selected_clock_name"] == "any_handoff"
    assert lineage["disclosed_global_rows"] == 66
    assert lineage["identity_key"] == ["source_day", "side", "decision_time"]
    assert lineage["candidate_decision_equals_predecessor_entry"] is True
    assert lineage["candidate_entry_and_exit_shift_bars"] == 1
    assert lineage["clock"]["header_sha256"] == (
        "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495"
    )


def test_controls_and_support_gates_are_singleton_and_fail_closed() -> None:
    payload = p.build_manifest()
    controls = payload["source_only_controls"]
    support = payload["source_support_gate"]
    assert controls["ordered"] == [
        "primary",
        "handoff_without_price_lag",
        "price_lag_without_handoff",
        "fixed_noon",
        "stale_24h",
        "direction_flip",
        "anchor_side_year_permutation",
        "anchor_return_year_permutation",
        "deterministic_random_side",
    ]
    assert controls["permutation_rng"] is None
    assert support["train_events_min"] == 24
    assert support["selection_events_min"] == 12
    assert support["permutation_exact_entry_jaccard_max"] == 0.35
    assert support["permutation_same_side_reproduction_max"] == 0.60
    assert "before comparators/outcomes" in support["failure_action"]


def test_common_window_and_comparator_parsers_are_frozen() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    assert novelty["common_window_policy_sha256"] == p.COMMON_WINDOW_POLICY_SHA256
    assert novelty["raw_validation_before_filter"] is True
    assert novelty["minimum_selected_contained_rows"] == 10
    comparators = {item["id"]: item for item in novelty["comparators"]}
    assert list(comparators) == [
        "IVLIR-72",
        "BAFR-24F",
        "AFCS-144",
        "LVRT-R0",
        "SMCC-144",
        "QLCD-288",
    ]
    assert comparators["BAFR-24F"]["tolerant_containment"] is False
    assert comparators["IVLIR-72"]["side_encoding"] == {
        "LONG": "LONG",
        "SHORT": "SHORT",
    }
    assert comparators["BAFR-24F"]["side_encoding"] == {
        "1": "LONG",
        "-1": "SHORT",
    }
    assert comparators["LVRT-R0"]["hold_bars"] == 12
    assert comparators["SMCC-144"]["hold_bars"] == 144
    assert comparators["QLCD-288"]["hold_bars"] == 288
    assert comparators["IVLIR-72"]["source_day_column"] == "source_day"


def test_economics_stages_and_llm_non_rescue_are_frozen() -> None:
    payload = p.build_manifest()
    economics = payload["economic_gates"]
    sequence = payload["strict_sequence"]
    llm = payload["llm_boundary"]
    assert economics["each_stage_cagr_to_strict_mdd_min"] == 3.0
    assert economics["each_stage_strict_mdd_pct_max"] == 15.0
    assert economics["mean_gross_underlying_bp_min"] == 15.0
    assert economics["component_margin_bp_min"] == 5.0
    assert sequence["stop_at_first_failure"] is True
    assert sequence["no_parameter_repair"] is True
    assert llm["action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert llm["protocol_freeze_after_base_train_before_selection"] is True
    assert llm["selection_feedback_forbidden"] is True
    assert "side" in llm["may_not_change"]


def test_frozen_dependency_hashes_match_without_decoding_rows() -> None:
    dependencies = p.frozen_dependencies()
    assert dependencies[p.MECHANISM_DOCUMENT] == p.MECHANISM_DOCUMENT_SHA256
    assert dependencies[p.MARKET_SOURCE] == p.MARKET_SOURCE_SHA256
    assert len(dependencies) == 14
    p.validate_frozen_dependencies()
    predecessor = p._predecessor_contract()["clock"]
    assert p.sha256_csv_header(predecessor["path"]) == predecessor["header_sha256"]


def test_write_once_is_reproducible_and_rejects_drift(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    first = p.build_manifest()
    second = p.build_manifest()
    assert first == second
    assert "created_at" not in first
    assert p.write_once(output, first) == "created"
    stored = json.loads(output.read_text())
    p.validate_manifest(stored)
    assert first["manifest_hash"] == second["manifest_hash"]
    assert p.write_once(output, second) == "verified_existing"
    stored["policy"]["hold_bars"] = 71
    with pytest.raises(RuntimeError, match="hash mismatch"):
        p.validate_manifest(stored)


def test_validate_manifest_rejects_self_rehashed_non_policy_drift() -> None:
    payload = p.build_manifest()
    payload["novelty_contract"]["minimum_selected_contained_rows"] = 11
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)
