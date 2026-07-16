from __future__ import annotations

from training import preregister_cross_venue_radial_refill_compression as crrc


def test_protocol_is_a_deterministic_singleton() -> None:
    first = crrc.protocol()
    second = crrc.protocol()
    assert first == second
    assert crrc.canonical_hash(first) == crrc.canonical_hash(second)
    assert first["selection_2023"]["singleton_no_parameter_ranking"] is True


def test_support_selection_is_outcome_blind_and_deterministic() -> None:
    frozen = crrc.protocol()
    selection = frozen["support_selection"]
    assert selection["candidate_cells_inspected"] == 14
    assert selection["selection_used_outcomes"] is False
    assert "selected cell" in frozen["evidence_boundary"][
        "incidence_correction_before_outcomes"
    ]
    assert selection["selected_cell"] == {
        "q_add": 0.85,
        "q_withdraw": 0.75,
        "q_net": 0.55,
        "q_flicker": 0.85,
        "observed_nonoverlap_events": 156,
    }
    assert frozen["evidence_boundary"]["post_entry_price_return_or_equity_opened"] is False


def test_feature_clock_excludes_current_and_future_rows() -> None:
    frozen = crrc.protocol()
    quantile = frozen["feature_formula"]["lagged_quantile"]
    assert quantile["window_rows"] == 8640
    assert quantile["minimum_finite_prior_rows"] == 6912
    assert quantile["shift_rows"] == 1
    assert quantile["current_row_excluded"] is True
    assert frozen["clock"]["feature_available_time"] == "t+5m"
    assert frozen["clock"]["entry_time"].startswith("t+10m open")


def test_formula_has_no_division_or_ambiguous_conflict() -> None:
    frozen = crrc.protocol()
    formula = frozen["feature_formula"]
    assert "never division" in formula["per_venue_and_side"]["inner_flicker"]
    assert formula["long"] == "bid_both AND NOT ask_both"
    assert formula["short"] == "ask_both AND NOT bid_both"
    assert formula["conflict"].endswith("is flat")


def test_strict_execution_and_live_parity_are_mandatory() -> None:
    frozen = crrc.protocol()
    assert frozen["universe"]["execution_instrument"].endswith("perpetual only")
    assert frozen["execution"]["base_cost_bp_per_notional_side"] == 6.0
    assert frozen["execution"]["strict_mdd"].startswith("global/pre-entry HWM")
    assert frozen["execution"]["cagr"].startswith("full declared wall-clock")
    assert frozen["live_parity_contract"]["archive_is_not_live"] is True
    assert "live UM and CM local books" in frozen["live_parity_contract"]["collector_required"]


def test_support_and_sequential_oos_fail_closed() -> None:
    frozen = crrc.protocol()
    gate = frozen["support_gate"]
    assert gate["nonoverlap_events_at_least"] == 150
    assert gate["events_each_quarter_at_least"] == 25
    assert frozen["outcome_blind_independence_gate"]["post_entry_return_or_pnl_forbidden"] is True
    assert frozen["sequential_oos"]["2024_opened_only_after_complete_2023_pass"] is True
    assert frozen["sequential_oos"]["2025_opened_only_after_complete_2024_pass"] is True
    assert frozen["sequential_oos"]["2026_opened_only_after_complete_2025_pass"] is True
    assert "Retire without repair" in frozen["stop_rule"]
