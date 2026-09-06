from training import preregister_high_volatility_bito_futures_flow_sponsorship_relay as prereg


def test_frozen_singleton_policy():
    value = prereg.build()
    assert value["policy_id"] == "HVBITOFL-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["policy"]["relative_volume_rank_min"] == 0.65
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["research_boundary"]["bito_values_opened"] is False
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_canonical_hash():
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
