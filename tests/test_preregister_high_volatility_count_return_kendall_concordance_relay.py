from training import preregister_high_volatility_count_return_kendall_concordance_relay as s


def test_blind_canonical_registration():
    payload = s.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_fixed_kendall_contract():
    payload = s.build()
    assert payload["policy_id"] == "HVCRKC-8"
    assert payload["policy"]["decision_hours_utc"] == [2, 10, 18]
    assert payload["policy"]["strength_rank_min"] == 0.75
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert not payload["research_boundary"]["repair_of_prior_candidate"]
