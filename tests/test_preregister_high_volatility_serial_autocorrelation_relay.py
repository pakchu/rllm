from training import preregister_high_volatility_serial_autocorrelation_relay as prereg


def test_manifest_is_deterministic_and_valid() -> None:
    first = prereg.build()
    second = prereg.build()
    assert first == second
    prereg.validate(first)
    assert first["manifest_hash"] == second["manifest_hash"]


def test_protocol_freezes_required_boundaries() -> None:
    payload = prereg.build()
    assert payload["policy_id"] == "HVSAR-12"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["features"]["grid"] is False
    assert payload["policy"] == {
        "return_bars": 144,
        "prior_boundaries": 270,
        "prior_min_boundaries": 252,
        "variation_rank_min": 0.70,
        "absolute_autocorrelation_rank_min": 0.75,
        "entry_delay_minutes": 5,
        "hold_hours": 12,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }
    assert payload["clock"]["side"] == "sign(completed_return)*sign(lag_one_autocorrelation)"
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
    assert "no path, rank, sign, hold, clock, subset, threshold, or control repair" in payload["stopping_rule"]
