from training import preregister_high_volatility_trade_arrival_memory_continuation as prereg


def test_preregistration_is_singleton_blind_and_terminal():
    value = prereg.build(); prereg.validate(value)
    assert value["policy_id"] == "HVTAMC-8"
    assert value["singleton"] is True
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["research_boundary"]["candidate_count"] == 1
    assert value["stopping_rule"].startswith("terminal first failure")


def test_policy_and_gates_are_frozen():
    value = prereg.build()
    assert value["policy"]["memory_rank_min"] == 0.75
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["source_support_gates"]["minimum_events"] == {"train": 8, "test": 12, "eval": 12, "final": 8}
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
