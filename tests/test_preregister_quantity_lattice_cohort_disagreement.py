from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_quantity_lattice_cohort_disagreement as qlcd


def test_manifest_is_singleton_outcome_blind_and_price_free() -> None:
    payload = qlcd.build_manifest()
    qlcd.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["policy"]["policy_id"] == "QLCD-288"
    assert payload["feature_contract"]["candidate_count"] == 1
    assert payload["feature_contract"]["price_fields_used"] is False
    assert payload["feature_contract"]["notional_fields_used"] is False


def test_policy_freezes_lattice_threshold_and_execution() -> None:
    payload = qlcd.build_manifest()
    policy = payload["policy"]
    feature = payload["feature_contract"]
    assert feature["coarse"].startswith("quantity_mbtc % 100")
    assert feature["fine"].startswith("quantity_mbtc % 10")
    assert policy["score_quantile"] == 0.9975
    assert policy["execution_delay_bars"] == 2
    assert policy["hold_bars"] == 288
    assert feature["threshold_grid"] is False


def test_manifest_binds_source_quarantine_and_prior_clocks() -> None:
    payload = qlcd.build_manifest()
    source = payload["source_contract"]
    assert source["archive_manifest_sha256"] == qlcd.ARCHIVE_MANIFEST_SHA256
    assert source["source_audit_sha256"] == qlcd.SOURCE_AUDIT_SHA256
    assert source["required_source_gap_days"][0] == "2020-01-15"
    registry = payload["novelty_gates"]["comparator_registry"]
    assert [item["family"] for item in registry] == [
        "MFIC",
        "AFCS",
        "TAAR",
        "RIFT",
        "PCP",
        "SMCC",
    ]
    assert all(len(item["sha256"]) == 64 for item in registry)


def test_manifest_hash_detects_mutation() -> None:
    payload = qlcd.build_manifest()
    payload["policy"]["hold_bars"] = 144
    with pytest.raises(RuntimeError, match="hash mismatch"):
        qlcd.validate_manifest(payload)


def test_write_once_refuses_changed_policy(tmp_path: Path) -> None:
    path = tmp_path / "qlcd.json"
    assert qlcd.write_manifest_once(path, qlcd.build_manifest()) == "created"
    assert qlcd.write_manifest_once(path, qlcd.build_manifest()) == "verified_existing"
    changed = json.loads(path.read_text())
    changed["policy"]["hold_bars"] = 144
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = qlcd.canonical_hash(core)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="frozen policy differs"):
        qlcd.write_manifest_once(path, qlcd.build_manifest())


def test_repository_artifact_matches_code() -> None:
    artifact = json.loads(Path(qlcd.DEFAULT_OUTPUT).read_text())
    qlcd.validate_manifest(artifact)
    assert artifact["manifest_hash"] == qlcd.build_manifest()["manifest_hash"]
