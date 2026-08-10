from training import preregister_high_volatility_mass_index_bulge_reversal as prereg


def test_boundary_is_outcome_blind_singleton():
    value = prereg.build()
    assert value["policy_id"] == "HVMIB-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["ema_periods"] == 9
    assert value["policy"]["mass_sum_periods"] == 25
    assert value["policy"]["bulge_arm_level"] == 27.0
    assert value["policy"]["bulge_release_level"] == 26.5
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_manifest_hash():
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
