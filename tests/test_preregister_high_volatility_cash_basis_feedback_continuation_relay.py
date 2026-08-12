from training import preregister_high_volatility_cash_basis_feedback_continuation_relay as p


def test_singleton_and_boundary():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVCBFC-8"
    assert value["source_incidence_opened"] is False
    assert value["research_boundary"]["grid"] is False


def test_frozen_gates_and_lag_operator():
    value = p.build()
    assert value["policy"]["feedback_rank_min"] == 0.80
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert "basis_change[t+1]" in value["features"]["cash_basis_feedback"]
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
