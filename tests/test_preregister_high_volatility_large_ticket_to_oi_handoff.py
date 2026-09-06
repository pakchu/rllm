from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_cross_structure_action_vote as hvcav
from training import preregister_high_volatility_large_ticket_to_oi_handoff as p


def test_ordered_singleton_and_frozen_thresholds() -> None:
    value = p.build(); p.validate(value)
    assert value["policy_id"] == "HVLTOIH-8"
    assert value["candidate_family"] == ["HVLTOIH-8"]
    assert value["policy"]["handoff_lag_hours"] == 3
    assert value["policy"]["coactivity_rank_min"] == 0.75
    assert value["policy"]["clustering_rank_min"] == 0.80
    assert value["policy"]["hold_hours"] == 8


def test_contract_gates_are_exact() -> None:
    value = p.build(); prior = hvcav.build()
    for key in ("stages", "source_support_gates", "gross9_novelty_gates", "economic_gates"):
        assert value[key] == prior[key]


def test_temporal_order_and_outcome_boundary() -> None:
    value = p.build()
    assert value["features"]["lagged_ticket_state"].startswith("at D-3h")
    assert value["features"]["current_oi_state"].startswith("at D")
    assert value["source_incidence_opened"] is False
    assert value["outcomes_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_prior_failure_disclosed_without_repair() -> None:
    boundary = p.build()["research_boundary"]
    assert boundary["prior_simultaneous_HVOILTJ_train_failure_known"] is True
    assert boundary["same_mechanism_as_HVOILTJ"] is False
    assert boundary["exact_ordered_handoff_incidence_or_outcomes_known"] is False
    assert boundary["prior_outcomes_used_to_set_lag_thresholds_side_hold_or_clock"] is False
    assert boundary["repair_of_prior_candidate"] is False


def test_manifest_drift_rejected() -> None:
    value = copy.deepcopy(p.build()); value["policy"]["handoff_lag_hours"] = 6
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
