from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_cross_structure_action_vote as hvcav
from training import preregister_high_volatility_spot_led_adverse_funding_continuation as p


def test_singleton_and_fixed_policy() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVSLAF-8"
    assert value["candidate_family"] == ["HVSLAF-8"]
    assert value["policy"] == {
        "history_cycles": 270,
        "minimum_history_cycles": 180,
        "path_minutes": 480,
        "variation_rank_min": 0.60,
        "entry_delay_minutes": 5,
        "hold_hours": 8,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.0010,
    }


def test_contract_gates_are_exact() -> None:
    value = p.build()
    prior = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        assert value[key] == prior[key]


def test_causal_multi_condition_boundary() -> None:
    value = p.build()
    features = value["features"]
    assert features["variation_rank"].startswith("strict-prior")
    assert "abs(spot_return)>abs(perpetual_return)" in features["cash_leadership"]
    assert "opposite sign" in features["funding_adversity"]
    assert features["side"] == "strict sign of spot_return"
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_prior_knowledge_disclosed_without_repair() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_spot_perpetual_OI_state_outcomes_known"] is True
    assert boundary["prior_funding_and_delayed_cash_source_failure_known"] is True
    assert boundary["exact_spot_led_adverse_funding_flow_conjunction_incidence_or_outcomes_known"] is False
    assert boundary["user_authorized_multi_condition_composition"] is True
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["prior_outcomes_used_to_set_rank_side_hold_or_clock"] is False


def test_manifest_drift_rejected() -> None:
    value = copy.deepcopy(p.build())
    value["policy"]["variation_rank_min"] = 0.55
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
