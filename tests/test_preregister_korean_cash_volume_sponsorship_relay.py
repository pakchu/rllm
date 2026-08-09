from training import preregister_korean_cash_volume_sponsorship_relay as prereg


def test_manifest_is_deterministic_and_valid() -> None:
    first = prereg.build(); second = prereg.build()
    assert first == second
    prereg.validate(first)


def test_protocol_freezes_independent_volume_sponsorship_rule() -> None:
    payload = prereg.build()
    assert payload["policy_id"] == "KCVSR-12"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["policy"]["upbit_volume_rank_min"] == 0.75
    assert payload["policy"]["binance_variation_rank_min"] == 0.65
    assert payload["clock"]["entry"] == "exact 08:05 UTC BTCUSDT perpetual open"
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
