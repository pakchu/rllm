from training import preregister_high_volatility_range_action_verification_ignition_relay as prereg


def test_boundary() -> None:
    value = prereg.build()
    assert value["policy_id"] == "HVRAVI-24" and value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False and value["gross9_rows_opened"] is False and value["singleton"] is True
    assert value["policy"]["fast_periods"] == 7 and value["policy"]["slow_periods"] == 65 and value["policy"]["trend_threshold_pct"] == 3
    assert value["research_boundary"]["grid"] is False and value["research_boundary"]["repair_of_prior_candidate"] is False and value["research_boundary"]["promoted_prior_control"] is False


def test_hash() -> None:
    value = prereg.build(); core = {key: item for key, item in value.items() if key != "manifest_hash"}; assert value["manifest_hash"] == prereg.canonical_hash(core)
