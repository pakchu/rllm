from training import preregister_high_volatility_directional_tail_index_asymmetry_relay as p


def test_boundary():
    value = p.build()
    assert value["policy_id"] == "HVDTIAR-6"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["singleton"] is True
    assert value["research_boundary"]["candidate_count"] == 1
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["policy"]["hill_k"] == 24
    assert value["policy"]["tail_asymmetry_rank_min"] == 0.60
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["policy"]["hold_hours"] == 6


def test_hash():
    value = p.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == p.canonical_hash(core)
