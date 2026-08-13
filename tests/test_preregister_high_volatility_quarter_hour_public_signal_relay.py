from training import preregister_high_volatility_quarter_hour_public_signal_relay as prereg


def test_public_signal_contract_is_causal_singleton_and_novelty_first() -> None:
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVQHPS-12"
    assert value["policy"]["technical_indicator_count"] == 28
    assert value["policy"]["quarter_hour_lags"] == 12
    assert value["policy"]["public_strength_rank_min"] == 0.95
    assert value["clock"]["hold"] == "12 elapsed hours"
    assert "[T-30m,T-15m)" in value["features"]["indicator_source"]
    assert "(close-lower)/lower" in value["features"]["volatility_4"]
    assert "adjust=False" in value["features"]["smoothing_conventions"]
    assert value["novelty_gates"]["must_pass_before_economics"] is True
    assert value["economic_gates"]["stop_on_first_failure"] is True
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
    assert value["research_boundary"]["candidate_incidence_opened"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_manifest_hash_is_canonical() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
