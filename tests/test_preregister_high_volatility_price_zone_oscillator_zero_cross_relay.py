from training import preregister_high_volatility_price_zone_oscillator_zero_cross_relay as prereg


def test_boundary_and_singleton():
    value = prereg.build()
    assert value["policy_id"] == "HVPZO-6"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["ema_periods"] == 14
    assert value["policy"]["hold_hours"] == 6
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_canonical_hash():
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
