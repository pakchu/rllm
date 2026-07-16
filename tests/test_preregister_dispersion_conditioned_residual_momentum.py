from __future__ import annotations

from training import preregister_dispersion_conditioned_residual_momentum as dcrm


def test_protocol_is_a_deterministic_singleton() -> None:
    first = dcrm.protocol()
    second = dcrm.protocol()
    assert first == second
    assert dcrm.canonical_hash(first) == dcrm.canonical_hash(second)
    assert first["selection_2023"]["singleton_no_parameter_ranking"] is True


def test_protocol_discloses_outcome_blind_support_choice() -> None:
    frozen = dcrm.protocol()
    boundary = frozen["evidence_boundary"]
    assert boundary["outcome_blind_weekly_feature_clock_inspected"] is True
    assert boundary["post_entry_returns_or_equity_opened"] is False
    assert len(boundary["support_only_scale_candidates_inspected"]) == 2
    assert "92 support events" in boundary["selection_basis"]


def test_feature_clock_has_no_current_or_future_bar() -> None:
    frozen = dcrm.protocol()
    formula = frozen["feature_formula"]
    assert formula["last_observable_bar"].startswith("Sunday 23:55")
    assert "timestamp >= decision_time is forbidden" in formula["row_cutoff"]
    assert formula["beta"]["shift_completed_hours"] == 1
    assert "symbol-specific factor_30d" in formula["score"]
    assert formula["dispersion_reference"]["current_state_excluded"] is True
    assert formula["dispersion_reference"]["quantile_interpolation"] == "linear"
    assert frozen["clock"]["entry_time"] == "Monday 00:05 UTC open"


def test_protocol_is_structurally_distinct_and_strict() -> None:
    frozen = dcrm.protocol()
    novelty = frozen["novelty_boundary"]
    assert "no OI" in novelty["versus_oi_funding_premium_kimchi"]
    assert "LORE/LORC" in novelty["not_globally_orthogonal"]
    assert novelty["llm_or_tree_dependency"] is False
    assert frozen["execution"]["base_cost_bp_per_notional_side"] == 6.0
    assert frozen["execution"]["strict_mdd"].startswith("global/pre-entry HWM")
    assert frozen["execution"]["cagr"].startswith("full declared wall-clock")
    assert frozen["outcome_blind_overlap_before_outcomes"][
        "post_entry_return_or_pnl_forbidden"
    ] is True


def test_holdouts_stop_after_first_failed_window() -> None:
    frozen = dcrm.protocol()
    assert frozen["sequential_oos"]["2024_opened_only_after_2023_pass"] is True
    assert frozen["sequential_oos"]["2025_opened_only_after_2024_pass"] is True
    assert frozen["sequential_oos"]["2026_opened_only_after_2025_pass"] is True
    assert "Open 2023 once only" in frozen["stop_rule"]
