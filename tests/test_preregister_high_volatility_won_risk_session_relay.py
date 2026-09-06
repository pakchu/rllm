from training import preregister_high_volatility_won_risk_session_relay as subject


def test_manifest_is_self_consistent_and_outcome_blind() -> None:
    payload = subject.build()
    subject.validate(payload)
    assert payload["policy_id"] == "HVWRSR-8"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["source_plan"]["fx"]["symbol"] == "USDKRW"
    assert payload["clock"]["entry"] == "exact BTCUSDT perpetual D 08:05 UTC open"
    assert payload["policy"]["hold_hours"] == 8
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True


def test_required_gates_are_frozen() -> None:
    payload = subject.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert payload["source_support_gates"]["minority_side_share_min"] == 0.2
    assert payload["source_support_gates"]["max_month_share"] == 0.45
    assert payload["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
