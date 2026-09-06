from training import preregister_high_volatility_directional_movement_index_crossover_relay as prereg


def test_boundary() -> None:
    value = prereg.build()
    assert value["policy_id"] == "HVDMIC-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["directional_periods"] == 14
    assert value["policy"]["variation_rank_min"] == .65
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_hash() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
