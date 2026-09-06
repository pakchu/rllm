import json

from training import preregister_high_volatility_cross_alt_rank_persistence_relay as prereg


def test_manifest_is_deterministic_and_self_bound() -> None:
    payload = prereg.build()
    assert payload == prereg.build()
    prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_frozen_formulas_and_policy() -> None:
    payload = prereg.build()
    features = payload["features"]
    policy = payload["policy"]
    assert payload["policy_id"] == "HVCARP-8"
    assert payload["slug"] == "high_volatility_cross_alt_rank_persistence_relay"
    assert payload["as_of_date"] == "2026-08-10"
    assert features["universe"] == prereg.SYMBOLS
    assert "360 exact aligned unique" in features["aligned_window"]
    assert "last close/first open" in features["alt_first_half_return"]
    assert "last close/first open" in features["alt_second_half_return"]
    assert "ties invalidate" in features["strict_cross_section"]
    assert "1 - 6*sum_j((R1_j-R2_j)^2)/(6*(6^2-1))" in features["rank_persistence"]
    assert "current excluded" in features["rank_persistence_rank"]
    assert "sum of 360 squared" in features["btc_variation"]
    assert "strictly nonzero" in features["direction_confirmation"]
    assert policy == {
        "window_hours": 6,
        "half_window_minutes": 180,
        "cross_section_size": 6,
        "history_hours": 2160,
        "minimum_history_hours": 1440,
        "rank_persistence_rank_min": 0.80,
        "btc_variation_rank_min": 0.65,
        "entry_delay_minutes": 5,
        "hold_hours": 8,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }
    assert payload["clock"]["split_crossing_action"] == "skip"
    assert payload["clock"]["reservation"] == "global half-open; exit first on equal-time entry"
    assert payload["stages"] == {
        "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
    }
    assert payload["source_support_gates"] == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.20,
        "max_month_share": 0.45,
    }
    assert payload["novelty_gates"] == {
        "exact_entry_jaccard_max": 0.10,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_bar_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["strict_mdd_max_pct"] == 15.0
    assert payload["economic_gates"]["mean_gross_underlying_min_bp"] == 20.0
    assert payload["economic_gates"]["weekly_signflip_one_sided_p_max"] == 0.10
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
    assert payload["economic_gates"]["each_calendar_half_positive"] is True
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert payload["source_plan"]["bars"]["symbols"] == prereg.SYMBOLS


def test_frozen_boundary_and_control_flags() -> None:
    payload = prereg.build()
    boundary = payload["research_boundary"]
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert boundary["prior_cross_alt_breadth_residual_leadership_outcomes_known"] is True
    assert boundary["prior_btc_temporal_rank_outcomes_known"] is True
    assert boundary["repository_exact_cross_alt_two_half_rank_persistence_candidate_found"] is False
    assert boundary["prior_outcomes_used_to_set_formula_rank_side_hold_or_clock"] is False
    assert boundary["candidate_incidence_opened"] is False
    assert boundary["postentry_return_or_pnl_opened"] is False
    assert boundary["gross9_rows_opened"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False
    assert boundary["selection_basis"] == "independent cross-sectional leadership-order persistence"
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
    assert payload["diagnostic_controls"]["names"] == [
        "no_btc_variation_gate",
        "no_rank_persistence_gate",
        "second_half_rank_reversal",
        "one_hour_stale_features",
        "direction_flip",
        "forced_long",
    ]
    assert "rho<=-0.80" in payload["diagnostic_controls"]["definitions"][
        "second_half_rank_reversal"
    ]


def test_serialized_payload_round_trip(tmp_path) -> None:
    payload = prereg.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    loaded = json.loads(path.read_text())
    prereg.validate(loaded)
    assert loaded == payload
