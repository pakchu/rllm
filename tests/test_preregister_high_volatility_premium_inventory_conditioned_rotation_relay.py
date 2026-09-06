from training import preregister_high_volatility_premium_inventory_conditioned_rotation_relay as p


def test_hvpicr_contract_is_singleton_outcome_blind_and_strict():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVPICR-8"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["features"]["current_oi_block"].endswith("strict nonzero current log change")
    assert value["mechanism"]["side"].startswith("product of the strict current premium")
    assert "when OI expands" in value["clock"]["side"] and "when OI contracts" in value["clock"]["side"]
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["economic_gates"]["stop_on_first_failure"] is True
