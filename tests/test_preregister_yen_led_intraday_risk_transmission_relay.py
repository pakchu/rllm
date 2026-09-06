from training import preregister_yen_led_intraday_risk_transmission_relay as prereg


def test_ylirtr_is_outcome_blind_independent_singleton():
    payload = prereg.build()
    assert payload["policy_id"] == "YLIRTR-12"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["singleton"] is True
    boundary = payload["research_boundary"]
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["prior_fx_event_sets_reused"] is False


def test_ylirtr_freezes_lead_coupling_and_strict_gates():
    payload = prereg.build(); policy = payload["policy"]
    assert policy["prior_sessions"] == 252
    assert policy["minimum_prior_sessions"] == 126
    assert policy["lead_correlation_rank_min"] == 0.70
    assert policy["realized_variation_rank_min"] == 0.65
    assert policy["hold_hours"] == 12
    assert payload["clock"]["decision"] == "exact weekday D 21:00 UTC after the FX session and BTC variation are complete"
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_ylirtr_hash_binds_core():
    payload = prereg.build()
    assert payload["manifest_hash"] == prereg.canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
