from __future__ import annotations

from training import preregister_post_funding_crowding_release_episode_v2 as pfcr2


def test_pfcr2_protocol_is_deterministic_singleton() -> None:
    first = pfcr2.protocol()
    second = pfcr2.protocol()
    assert first == second
    assert pfcr2.canonical_hash(first) == pfcr2.canonical_hash(second)
    assert first["selection_2023_2024"]["singleton_no_parameter_ranking"] is True


def test_pfcr2_discloses_outcome_blind_derivation() -> None:
    frozen = pfcr2.protocol()
    derivation = frozen["protocol_derivation"]
    assert derivation["parent_post_entry_returns_calculated"] is False
    assert derivation["outcome_or_post_entry_price_used"] is False
    assert derivation["selected_cooldown_hours"] == 36
    assert derivation["new_protocol_not_parent_repair"] is True
    assert len(derivation["support_only_candidates_inspected"]) == 9


def test_pfcr2_clock_and_sequential_holdouts_are_frozen() -> None:
    frozen = pfcr2.protocol()
    assert frozen["clock"]["episode_cooldown_hours"] == 36
    assert frozen["clock"]["cooldown_anchor"] == "previous accepted settlement timestamp"
    assert frozen["evidence_boundary"]["pfcr1_or_pfcr2_post_entry_returns_opened"] is False
    assert frozen["evidence_boundary"]["2025_eval_opened"] is False
    assert frozen["evidence_boundary"]["2026_holdout_opened"] is False


def test_pfcr2_keeps_parent_execution_and_orthogonality_contracts() -> None:
    frozen = pfcr2.protocol()
    assert frozen["execution"]["base_cost_bp_per_notional_side"] == 6.0
    assert frozen["execution"]["strict_mdd"].startswith("global/pre-entry HWM")
    assert frozen["orthogonality_after_standalone_pass"][
        "absolute_daily_pnl_pearson_at_most"
    ] == 0.30
