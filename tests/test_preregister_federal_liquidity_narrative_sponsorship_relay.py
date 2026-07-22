from __future__ import annotations

import json
from pathlib import Path

from training import preregister_federal_liquidity_narrative_sponsorship_relay as p


ARTIFACT = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_"
    "preregistration_2026-07-23.json"
)


def test_manifest_is_outcome_and_incidence_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "FLNSR-2016"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["causal_feature_contract"]["future_source_row_used"] is False
    assert payload["causal_feature_contract"]["btc_market_field_used"] is False


def test_policy_freezes_singleton_conjunction_and_execution() -> None:
    payload = p.build_manifest()
    policy = payload["policy"]
    assert policy["liquidity_delta_releases"] == 1
    assert policy["liquidity_rank_lookback_releases"] == 104
    assert (
        policy["liquidity_lower_rank_numerator"],
        policy["liquidity_upper_rank_numerator"],
    ) == (83, 125)
    assert policy["narrative_recent_days"] == 7
    assert policy["narrative_baseline_days"] == 21
    assert policy["hold_bars"] == 2_016
    feature = payload["causal_feature_contract"]
    assert "same non-neutral side" in feature["primary"]
    assert payload["execution_contract"]["entry"].endswith("plus ten minutes")


def test_ancestor_outcome_seen_boundary_is_explicit() -> None:
    boundary = p.build_manifest()["research_history_boundary"]
    assert boundary["flcc_train_2020_2022_outcomes_seen_and_rejected"] is True
    assert boundary["gnrc_train_and_2023_outcomes_seen_and_rejected"] is True
    assert boundary["exact_flnsr2016_clock_or_outcomes_seen"] is False
    assert "not a pristine" in boundary["claim_scope"]


def test_controls_and_llm_cannot_repair_base_event() -> None:
    payload = p.build_manifest()
    assert payload["source_support_gate"]["controls_affect_primary_pass"] is False
    llm = payload["llm_boundary"]
    assert llm["allowed_only_after_standalone_train_and_selection_pass"]
    assert llm["requires_separate_overlay_preregistration"]
    assert llm["later_action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert {"candidate clock", "side", "entry", "hold"}.issubset(
        set(llm["llm_may_not_change"])
    )


def test_flcc_clock_novelty_and_singleton_multiplicity_are_frozen() -> None:
    payload = p.build_manifest()
    novelty = payload["source_only_novelty_gate"]
    assert novelty["comparator_sha256"].startswith("7ebb0450")
    assert novelty["jaccard_max_each_flcc_candidate"] == 0.50
    assert novelty["flnsr_containment_max_each_flcc_candidate"] == 0.70
    assert novelty["comparator_outcomes_read"] is False
    assert payload["economic_gates"]["monthly_cluster_signflip_p_max"] == 0.05
    assert "singleton" in payload["economic_gates"]["multiplicity_scope"]


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
    assert stored["manifest_hash"] == p.build_manifest()["manifest_hash"]
