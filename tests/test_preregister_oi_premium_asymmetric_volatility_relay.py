from training import preregister_oi_premium_asymmetric_volatility_relay as prereg


def test_oipar_is_outcome_blind_frozen_singleton():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "OIPAR-ASYM"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["singleton"] is True
    assert payload["frozen_states"]["no_threshold_side_hold_stride_tuning"] is True
    assert payload["research_boundary"]["combined_candidate_outcomes_known"] is False
    assert payload["research_boundary"]["grid"] is False
