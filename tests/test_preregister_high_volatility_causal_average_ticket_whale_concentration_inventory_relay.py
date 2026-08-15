from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_causal_average_ticket_whale_concentration_inventory_relay as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav


def test_singleton_mechanism_and_fixed_policy() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVCATWIR-8"
    assert value["candidate_family"] == ["HVCATWIR-8"]
    assert value["policy"] == {
        "history_cycles": 270,
        "minimum_history_cycles": 180,
        "path_minutes": 480,
        "half_minutes": 240,
        "average_ticket_acceleration_rank_min": 0.60,
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


def test_causal_average_ticket_boundary() -> None:
    value = p.build()
    features = value["features"]
    assert features["oi_expansion"].startswith("strict positive")
    assert features["average_ticket_acceleration_rank"].startswith("strict-prior")
    assert features["variation_rank"].startswith("strict-prior")
    assert "quote_asset_volume" in features["completed_cycle"]
    assert "number_of_trades" in features["completed_cycle"]
    assert "strict positive trade-arrival deceleration" in features["eligible"]
    assert features["trade_arrival_deceleration"].startswith("log(first-half")
    assert value["mechanism"]["side"].startswith("strict sign")
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_prior_failures_disclosed_without_repair() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_HVCATAIR_source_novelty_and_train_failure_known"] is True
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
