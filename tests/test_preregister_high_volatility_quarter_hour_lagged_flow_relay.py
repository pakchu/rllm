from training import preregister_high_volatility_quarter_hour_lagged_flow_relay as prereg


def test_frozen_contract_is_singleton_causal_and_novelty_first() -> None:
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVQHLF-4"
    assert value["policy"]["quarter_hour_lags"] == 12
    assert value["policy"]["ols_minimum_observations"] == 5760
    assert value["policy"]["flow_strength_rank_min"] == 0.95
    assert value["clock"]["hold"] == "4 elapsed hours"
    assert value["novelty_gates"]["must_pass_before_economics"] is True
    assert value["economic_gates"]["stop_on_first_failure"] is True
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
    assert value["research_boundary"]["candidate_incidence_opened"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_manifest_hash_uses_frozen_canonical_json() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
