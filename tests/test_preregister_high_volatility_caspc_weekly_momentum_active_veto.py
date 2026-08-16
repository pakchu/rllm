from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_caspc_weekly_momentum_active_veto as p


def test_manifest_freezes_cross_timescale_singleton_before_incidence() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVCASPCWMV-8"
    assert value["action_ids"] == ["HVCASPC-8", "HVWMR-72"]
    assert value["candidate_family"] == ["HVCASPCWMV-8"]
    assert value["single_candidate_only"] is True
    assert value["combined_incidence_opened"] is False
    assert value["combined_outcomes_opened"] is False


def test_opposite_active_weekly_state_is_the_only_veto() -> None:
    assert p.active_state_veto_side(1, 0) == 1
    assert p.active_state_veto_side(1, 1) == 1
    assert p.active_state_veto_side(-1, -1) == -1
    assert p.active_state_veto_side(1, -1) == 0
    assert p.active_state_veto_side(-1, 1) == 0
    with pytest.raises(ValueError, match="exact -1, 0, or \\+1"):
        p.active_state_veto_side(1, True)


def test_no_threshold_weight_or_subset_repair_is_authorized() -> None:
    value = p.build()
    construction = value["construction"]
    assert construction["primary_action"] == "HVCASPC-8"
    assert construction["veto_action"] == "HVWMR-72"
    assert construction["additional_or_tuned_thresholds"] == "none"
    assert construction["weights"] == "none"
    assert construction["alternatives"] == "none"
    assert value["research_boundary"]["prior_active_veto_family_used_for_subset_or_threshold_repair"] is False


def test_component_artifacts_are_hash_pinned_and_non_economic() -> None:
    value = p.build()
    for artifacts in value["component_artifacts"].values():
        for kind, artifact in artifacts.items():
            assert p.sha256_file(artifact["path"]) == artifact["sha256"]
            assert "economic" not in artifact["path"]
            assert kind in {"preregistration", "support", "gross9", "clock"}


def test_manifest_drift_is_rejected() -> None:
    value = copy.deepcopy(p.build())
    value["construction"]["routing_rule"] = "use the weekly side"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(value)
