from training import preregister_high_volatility_wilder_adx_trend_relay as prereg


def test_boundary() -> None:
    value = prereg.build()
    assert value["policy_id"] == "HVWADX-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["adx_periods"] == 14
    assert value["policy"]["adx_min"] == 25
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_hash() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
