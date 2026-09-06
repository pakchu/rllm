from training import preregister_high_volatility_vix_transfer_entropy_forecast_relay as prereg


def test_contract_is_causal_singleton_and_novelty_first() -> None:
    value = prereg.build(); prereg.validate(value)
    assert value["policy_id"] == "HVVIXTE-24"
    assert value["policy"]["transition_history"] == 756
    assert value["policy"]["minimum_transitions"] == 252
    assert value["policy"]["minimum_conditioning_cell"] == 30
    assert value["policy"]["strength_rank_min"] == 0.75
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert "y completion time is <= current decision" in value["features"]["causal_history"]
    assert value["novelty_gates"]["must_pass_before_economics"] is True
    assert value["economic_gates"]["stop_on_first_failure"] is True
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
    assert value["research_boundary"]["candidate_incidence_opened"] is False


def test_manifest_hash_is_canonical() -> None:
    value = prereg.build(); core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
