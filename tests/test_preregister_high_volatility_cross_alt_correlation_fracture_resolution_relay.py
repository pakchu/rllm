import json

from training import (
    preregister_high_volatility_cross_alt_correlation_fracture_resolution_relay as prereg,
)


def test_manifest_is_deterministic_and_self_bound() -> None:
    payload = prereg.build()
    assert payload == prereg.build()
    prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_formulas_and_clock_are_frozen() -> None:
    payload = prereg.build()
    features = payload["features"]
    policy = payload["policy"]

    assert payload["policy_id"] == "HVCACFR-8"
    assert payload["slug"] == (
        "high_volatility_cross_alt_correlation_fracture_resolution_relay"
    )
    assert payload["as_of_date"] == "2026-08-10"
    assert payload["singleton"] is True
    assert features["universe"] == prereg.SYMBOLS
    assert features["decision_grid"] == "one exact daily decision D at 03:00 UTC"
    assert "720 exact aligned unique" in features["aligned_window"]
    assert features["minute_return"] == (
        "for each symbol and minute, natural log(close/open)"
    )
    assert "arithmetic mean of the third and fourth" in features[
        "median_alt_minute_return"
    ]
    assert "360 BTC minute returns" in features["first_half_correlation"]
    assert "[D-12h,D-6h)" in features["first_half_correlation"]
    assert "[D-6h,D)" in features["second_half_correlation"]
    assert "population variances" in features["first_half_correlation"]
    assert "strictly positive population variance" in features[
        "correlation_validity"
    ]
    assert features["correlation_fracture"] == (
        "first_half_correlation - second_half_correlation"
    )
    assert "at most 180" in features["correlation_fracture_rank"]
    assert "minimum 90" in features["correlation_fracture_rank"]
    assert "current excluded; rank>=0.75" in features[
        "correlation_fracture_rank"
    ]
    assert "sum of all 720 squared" in features["btc_variation"]
    assert "rank>=0.65" in features["btc_variation_rank"]
    assert "final 120" in features["direction"]
    assert "strictly nonzero" in features["direction"]
    assert features["btc_direction_confirmation"] is False
    assert "immediately previous exact source-valid daily decision" in features[
        "onset"
    ]

    assert policy == {
        "window_minutes": 720,
        "half_window_minutes": 360,
        "direction_window_minutes": 120,
        "cross_section_size": 6,
        "prior_valid_days": 180,
        "minimum_prior_valid_days": 90,
        "correlation_fracture_rank_min": 0.75,
        "btc_variation_rank_min": 0.65,
        "entry_delay_minutes": 5,
        "hold_hours": 8,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }
    assert payload["clock"]["entry"] == "exact BTCUSDT perpetual D+5m open"
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["clock"]["reservation"] == (
        "global half-open; exit first on equal-time entry"
    )
    assert payload["clock"]["split_crossing_action"] == "skip"
    assert payload["clock"]["gross_exposure"] == 0.5


def test_standard_gates_and_controls_are_frozen() -> None:
    payload = prereg.build()

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
    assert payload["post_stage_volatility_audit"]["prerequisite"] == (
        "unchanged candidate passes train, test, eval, and final"
    )
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False

    controls = payload["diagnostic_controls"]
    assert controls["names"] == [
        "no_btc_variation_gate",
        "no_correlation_fracture_gate",
        "contemporaneous_full_window_correlation",
        "one_day_stale_features",
        "direction_flip",
        "forced_long",
    ]
    assert controls["cannot_be_promoted"] is True
    full_window = controls["definitions"][
        "contemporaneous_full_window_correlation"
    ]
    assert "full 720 BTC and median-alt" in full_window
    assert "own strict-prior midrank" in full_window
    assert "rank>=0.75" in full_window


def test_research_and_source_boundaries_are_frozen() -> None:
    payload = prereg.build()
    boundary = payload["research_boundary"]

    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert boundary["prior_cross_alt_outcomes_known"] is True
    assert boundary["prior_eth_disagreement_outcomes_known"] is True
    assert boundary["prior_leadership_outcomes_known"] is True
    assert boundary[
        "repository_exact_basket_correlation_fracture_candidate_found"
    ] is False
    assert boundary["prior_outcomes_used_to_set_formula_rank_side_hold_or_clock"] is False
    assert boundary["candidate_incidence_opened"] is False
    assert boundary["postentry_return_or_pnl_opened"] is False
    assert boundary["gross9_rows_opened"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False
    assert payload["source_plan"]["bars"]["query_window"] == [
        "2023-01-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
    ]
    assert payload["source_plan"]["bars"]["table"] == "bars_binance"
    assert payload["source_plan"]["bars"]["interval"] == "1m"
    assert payload["source_plan"]["bars"]["symbols"] == prereg.SYMBOLS
    assert "no universe, basket, window" in payload["stopping_rule"]


def test_serialized_payload_round_trip(tmp_path) -> None:
    payload = prereg.build()
    path = tmp_path / "prereg.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    loaded = json.loads(path.read_text())
    prereg.validate(loaded)
    assert loaded == payload
