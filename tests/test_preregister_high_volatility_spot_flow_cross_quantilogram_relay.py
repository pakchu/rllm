from training import preregister_high_volatility_spot_flow_cross_quantilogram_relay as s


def test_blind_canonical_registration():
    result = s.build()
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.canonical_hash(core)
    assert not result["outcomes_opened"]
    assert not result["source_incidence_opened"]
    assert not result["gross9_rows_opened"]


def test_fixed_cross_quantilogram_contract():
    result = s.build()
    assert result["policy_id"] == "HVSFCQ-8"
    assert result["policy"]["tail_probability"] == 0.25
    assert result["policy"]["lag_minutes"] == 1
    assert result["policy"]["active_score_rank_min"] == 0.75
    assert result["policy"]["decision_hours_utc"] == [2, 10, 18]
    assert result["clock"]["hold"] == "8 elapsed hours"
    assert not result["research_boundary"]["repair_of_prior_candidate"]
