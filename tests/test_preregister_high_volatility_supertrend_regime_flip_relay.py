from training import preregister_high_volatility_supertrend_regime_flip_relay as prereg


def test_boundary_is_outcome_blind_singleton():
    value = prereg.build()
    assert value["policy_id"] == "HVSTR-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["atr_periods"] == 10
    assert value["policy"]["atr_multiplier"] == 3.0
    assert value["policy"]["initial_state"] == -1
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_manifest_hash():
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
