from training import preregister_high_volatility_spot_participation_sponsorship_relay as prereg


def test_manifest_is_deterministic_and_valid() -> None:
    first = prereg.build(); second = prereg.build()
    assert first == second
    prereg.validate(first)


def test_cash_share_rule_and_gates_are_frozen() -> None:
    payload = prereg.build()
    assert payload["policy_id"] == "HVSPSR-12"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["policy"]["spot_participation_rank_min"] == 0.75
    assert payload["policy"]["btc_variation_rank_min"] == 0.65
    assert payload["clock"]["entry"] == "exact BTCUSDT perpetual D 00:05 UTC open"
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
