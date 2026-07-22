from __future__ import annotations

import json
from pathlib import Path

from training import preregister_intrinsic_volume_flow_handoff_relay as p


ARTIFACT = Path(
    "results/intrinsic_volume_flow_handoff_relay_preregistration_2026-07-23.json"
)


def test_manifest_is_outcome_and_incidence_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "IVFHR-72"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["causal_feature_contract"]["future_bar_used_by_signal"] is False


def test_policy_freezes_handoff_price_lag_side_and_execution() -> None:
    payload = p.build_manifest()
    policy = payload["policy"]
    assert policy["intrinsic_volume_fraction"] == 0.50
    assert policy["current_flow_quantile"] == 0.60
    assert policy["prior_state_min_anchors"] == 3
    assert policy["entry_delay_bars"] == 1
    assert policy["hold_bars"] == 72
    feature = payload["causal_feature_contract"]
    assert "differs" in feature["handoff"]
    assert "<= 0" in feature["price_lag"]
    assert feature["side"].endswith("flow sign")
    assert "resets" in feature["invalid_anchor"]
    assert "calendar-consecutive" in feature["prior_state"]
    assert "bar-open" in payload["source_contract"]["timestamp_semantics"]
    assert payload["execution_contract"]["entry"].endswith("t+5min")
    assert payload["execution_contract"]["maximum_one_candidate_per_transition_episode"]


def test_source_seen_origin_and_no_repair_are_explicit() -> None:
    payload = p.build_manifest()
    boundary = payload["research_history_boundary"]
    assert boundary["ivlir_source_support_seen"] is True
    assert boundary["ivlir_post_entry_outcomes_seen"] is False
    assert "source-seen successor" in boundary["claim_scope"]
    assert payload["strict_sequence"]["no_parameter_repair"] is True


def test_llm_cannot_create_or_retime_trades() -> None:
    payload = p.build_manifest()
    boundary = payload["llm_boundary"]
    assert boundary["allowed_only_after_standalone_train_and_selection_pass"]
    assert boundary["later_action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert {"candidate clock", "side", "entry", "hold"}.issubset(
        set(boundary["llm_may_not_change"])
    )


def test_control_domains_and_component_margin_are_frozen() -> None:
    payload = p.build_manifest()
    support = payload["source_support_gate"]
    controls = payload["source_only_controls"]
    economics = payload["economic_gates"]
    assert support["controls_affect_primary_pass"] is False
    assert "split-contained" in support["gate_domain"]
    assert "consecutive accepted" in support["statistic_definitions"]["calendar_gap_days"]
    assert "cannot rescue or reject" in controls["shared_control_contract"]
    assert "90-anchor" in controls["shared_control_contract"]
    assert economics["component_control_universe"] == [
        "any_handoff",
        "no_price_lag",
        "no_flow_strength",
        "persistence_level",
        "fixed_noon_handoff",
    ]
    assert "before leverage" in economics["component_margin_statistic"]


def test_write_once_is_reproducible(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    first = p.build_manifest()
    assert p.write_once(output, first) == "created"
    stored = json.loads(output.read_text())
    p.validate_manifest(stored)
    second = p.build_manifest()
    assert first["manifest_hash"] == second["manifest_hash"]
    assert p.write_once(output, second) == "verified_existing"


def test_committed_artifact_matches_frozen_code() -> None:
    stored = json.loads(ARTIFACT.read_text())
    p.validate_manifest(stored)
    current = p.build_manifest()
    assert stored["manifest_hash"] == current["manifest_hash"]
