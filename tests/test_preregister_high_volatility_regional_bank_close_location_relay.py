from training import preregister_high_volatility_regional_bank_close_location_relay as prereg


def test_boundary_and_singleton():
    value = prereg.build()
    assert value["policy_id"] == "HVKRECLV-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["absolute_close_location_min"] == 0.5
    assert value["research_boundary"]["kre_values_opened"] is False
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_canonical_hash():
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
