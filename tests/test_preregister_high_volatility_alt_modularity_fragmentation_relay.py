from training import preregister_high_volatility_alt_modularity_fragmentation_relay as s


def test_blind_canonical_registration():
    result = s.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.canonical_hash(core)
    assert not result["outcomes_opened"]
    assert not result["source_incidence_opened"]
    assert not result["gross9_rows_opened"]


def test_fixed_modularity_contract():
    result = s.build()
    assert result["policy_id"] == "HVAMF-8"
    assert result["policy"]["alt_symbols"] == list(s.ALTS)
    assert result["policy"]["minimum_community_size"] == 2
    assert result["policy"]["modularity_rank_min"] == 0.75
    assert result["policy"]["decision_hours_utc"] == [4, 12, 20]
    assert result["clock"]["hold"] == "8 elapsed hours"
    assert not result["research_boundary"]["repair_of_prior_candidate"]
