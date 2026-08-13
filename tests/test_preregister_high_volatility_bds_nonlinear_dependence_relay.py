from training import preregister_high_volatility_bds_nonlinear_dependence_relay as s


def test_blind_canonical_registration():
    result = s.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.canonical_hash(core)
    assert not result["outcomes_opened"]
    assert not result["source_incidence_opened"]
    assert not result["gross9_rows_opened"]


def test_fixed_bds_contract():
    result = s.build()
    assert result["policy_id"] == "HVBDS-8"
    assert result["policy"]["embedding_dimension"] == 2
    assert result["policy"]["epsilon_standard_deviations"] == 0.5
    assert result["policy"]["departure_rank_min"] == 0.75
    assert result["policy"]["decision_hours_utc"] == [3, 11, 19]
    assert result["clock"]["hold"] == "8 elapsed hours"
    assert not result["research_boundary"]["repair_of_prior_candidate"]
