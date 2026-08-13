from training import preregister_high_volatility_spot_execution_activity_leadership_relay as s


def test_blind_canonical_registration():
    payload = s.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_fixed_execution_activity_leadership_contract():
    payload = s.build()
    assert payload["policy_id"] == "HVSEAL-8"
    assert payload["source_plan"]["spot"]["table"] == "bars_binance_spot"
    assert payload["source_plan"]["perpetual"]["table"] == "bars_binance"
    assert payload["policy"]["leadership_rank_min"] == 0.75
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert not payload["research_boundary"]["repair_of_prior_candidate"]
