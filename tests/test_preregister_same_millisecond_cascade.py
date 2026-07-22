from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_same_millisecond_cascade as smcc


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = smcc.build_manifest()
    smcc.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["policy"]["policy_id"] == "SMCC-144"
    assert payload["causal_feature_contract"]["candidate_count"] == 1
    assert payload["causal_feature_contract"]["threshold_or_hold_grid"] is False


def test_policy_freezes_exact_ms_collision_and_safe_execution() -> None:
    payload = smcc.build_manifest()
    policy = payload["policy"]
    feature = payload["causal_feature_contract"]
    assert feature["group"].startswith("exact equality")
    assert policy["minimum_group_agg_trade_count"] == 3
    assert policy["score_quantile"] == 0.995
    assert policy["execution_delay_bars"] == 2
    assert policy["hold_bars"] == 144


def test_manifest_binds_audited_archive_chain() -> None:
    source = smcc.build_manifest()["source_contract"]
    assert source["archive_manifest_sha256"] == smcc.ARCHIVE_MANIFEST_SHA256
    assert source["source_audit_sha256"] == smcc.SOURCE_AUDIT_SHA256
    assert source["raw_archives_persisted"] is False
    assert source["required_source_gap_days"][0] == "2020-01-15"
    assert source["smcc_specific_underlying_overlap_quarantine"]["2020-01-15"][
        "overlap_count"
    ] == 1


def test_manifest_freezes_comparators_controls_and_support_schema() -> None:
    payload = smcc.build_manifest()
    registry = payload["novelty_gates"]["comparator_registry"]
    assert [item["family"] for item in registry] == ["MFIC", "AFCS", "TAAR", "RIFT", "PCP"]
    assert all(len(item["sha256"]) == 64 for item in registry)
    assert len(payload["novelty_gates"]["dense_bafr"]["sha256"]) == 64
    controls = payload["falsification_controls"]
    assert "own prior-only q99.5 clock" in controls["remove_coherence"]
    assert "+288 bars" in controls["stale_twenty_four_hours"]
    support = payload["support_artifact_contract"]
    assert support["write_once"] is True
    assert support["outcomes_opened"] is False
    assert support["decision_values"] == ["PASS_SUPPORT", "REJECT_NO_REPAIR"]
    schedule = payload["support_schedule_contract"]
    assert "never cancels" in schedule["future_source_rule"]
    assert payload["novelty_gates"]["matching_algorithm"].startswith(
        "within each member coverage"
    )


def test_manifest_hash_detects_mutation() -> None:
    payload = smcc.build_manifest()
    payload["policy"]["hold_bars"] = 145
    with pytest.raises(RuntimeError, match="hash mismatch"):
        smcc.validate_manifest(payload)


def test_write_once_refuses_changed_policy(tmp_path) -> None:
    path = tmp_path / "smcc.json"
    assert smcc.write_manifest_once(path, smcc.build_manifest()) == "created"
    assert smcc.write_manifest_once(path, smcc.build_manifest()) == "verified_existing"
    changed = json.loads(path.read_text())
    changed["policy"]["hold_bars"] = 72
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = smcc.canonical_hash(core)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="frozen policy differs"):
        smcc.write_manifest_once(path, smcc.build_manifest())


def test_repository_preregistration_artifact_matches_code() -> None:
    artifact = json.loads(Path(smcc.DEFAULT_OUTPUT).read_text())
    smcc.validate_manifest(artifact)
    assert artifact["manifest_hash"] == smcc.build_manifest()["manifest_hash"]
    assert artifact["outcomes_opened"] is False
    assert artifact["source_incidence_opened"] is False
