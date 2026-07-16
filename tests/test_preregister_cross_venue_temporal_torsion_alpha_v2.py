from __future__ import annotations

import pytest

from training import preregister_cross_venue_temporal_torsion_alpha as v1
from training import preregister_cross_venue_temporal_torsion_alpha_v2 as v2


def test_v2_changes_only_declared_support_feasibility_fields() -> None:
    old = v1.build_manifest()
    new = v2.build_manifest()
    for key in (
        "economic_hypothesis",
        "source_contract",
        "policies",
        "route_rules",
        "selection_protocol",
        "execution_contract",
        "controls",
        "orthogonality_after_holdout",
    ):
        assert new[key] == old[key]
    assert new["outcomes_opened"] is False
    assert new["protocol_version"].endswith("v2")


def test_v2_separates_calendar_and_eligible_minima() -> None:
    feature = v2.build_manifest()["feature_contract"]
    assert "rolling_minimum_bars" not in feature
    assert feature["rolling_minimum_clean_calendar_bars"] == 2016
    assert feature["rolling_minimum_prior_route_events"] == 256


def test_v2_retains_global_quality_and_uses_declared_monthly_ceiling() -> None:
    support = v2.build_manifest()["support_freeze_before_returns"]
    assert support["global_missing_or_quarantined_fraction_max"] == 0.01
    assert support["monthly_missing_or_quarantined_fraction_max"] == 0.05


def test_v2_records_support_only_evidence_and_no_returns() -> None:
    payload = v2.build_manifest()
    repair = payload["support_only_feasibility_repair"]
    assert repair["parent_v1_outcomes_opened"] is False
    assert repair["parent_v1_support_manifest_hash"] == v2.PARENT_SUPPORT_MANIFEST_HASH
    assert "return" not in " ".join(repair["evidence_seen"])
    v2.validate_manifest(payload)


def test_tampering_fails_hash_validation() -> None:
    payload = v2.build_manifest()
    payload["feature_contract"]["rolling_minimum_prior_route_events"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        v2.validate_manifest(payload)
