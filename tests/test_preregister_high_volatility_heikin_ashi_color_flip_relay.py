from training import preregister_high_volatility_heikin_ashi_color_flip_relay as prereg


def test_boundary_is_outcome_blind_singleton():
    value = prereg.build()
    assert value["policy_id"] == "HVHA-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["initial_open_rule"] == "raw_open_close_midpoint"
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_manifest_hash():
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
