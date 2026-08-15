import copy

import pytest

from training import preregister_high_volatility_taker_flow_decay_acceptance_continuation as p


def test_outcome_blind_singleton_and_fixed_policy():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVTFDAC-8"
    assert value["candidate_family"] == ["HVTFDAC-8"]
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["policy"]["hold_hours"] == 8


def test_flow_decay_ordering_is_strict_and_causal():
    features = p.build()["features"]
    assert "0 < side*second_half_flow_imbalance" in features["directional_flow_decay"]
    assert "strict with no fitted threshold" in features["directional_flow_decay"]
    assert features["variation_rank"].startswith("strict-prior")
    assert "current excluded" in features["variation_rank"]


def test_manifest_drift_rejected():
    value = copy.deepcopy(p.build())
    value["policy"]["hold_hours"] = 12
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
