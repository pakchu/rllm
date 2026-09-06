from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_spot_lead_open_interest_acceleration_continuation as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav


def test_singleton_mechanism_and_fixed_policy() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVSPOIAC-8"
    assert value["candidate_family"] == ["HVSPOIAC-8"]
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


def test_exact_hvcav_stages_and_gates() -> None:
    value = p.build()
    prior = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        assert value[key] == prior[key]


def test_causal_spot_perpetual_oi_boundary() -> None:
    value = p.build()
    features = value["features"]
    assert features["oi_expansion"].startswith("strict positive")
    assert features["variation_rank"].startswith("strict-prior")
    assert "strict directional spot overshoot" in features["eligible"]
    assert features["directional_spot_overshoot"].startswith("sign(spot_return)")
    assert features["direction_agreement"].startswith("perpetual_return and spot_return")
    assert value["mechanism"]["side"].startswith("same strict sign")
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_prior_failures_disclosed_without_repair() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_spot_perpetual_family_failures_known"] is True
    assert boundary["prior_OI_state_family_failures_known"] is True
    assert boundary["exact_spot_lead_OI_expansion_incidence_or_outcomes_known"] is False
    assert boundary["prior_outcomes_informed_candidate_family"] is True
    assert boundary["prior_HVCVAIR_source_novelty_and_train_failure_known"] is True
    assert boundary["prior_HVCNAIR_source_novelty_and_train_failure_known"] is True
    assert boundary["same_mechanism_as_HVCVAIR"] is False
    assert boundary["same_mechanism_as_HVCNAIR"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["prior_outcomes_used_to_set_rank_side_hold_or_clock"] is False


def test_manifest_drift_rejected() -> None:
    value = copy.deepcopy(p.build())
    value["policy"]["variation_rank_min"] = 0.55
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
