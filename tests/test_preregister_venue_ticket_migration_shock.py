from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from training import preregister_venue_ticket_migration_shock as vtms


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = vtms.build_manifest()
    vtms.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(vtms.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]


def test_policy_uses_cross_venue_ticket_migration_and_safe_execution() -> None:
    payload = vtms.build_manifest()
    features = payload["causal_feature_contract"]
    execution = payload["execution_contract"]
    assert "spot_ticket" in features
    assert "perp_ticket" in features
    assert payload["policy"]["ticket_change_bars"] == 12
    assert payload["policy"]["execution_delay_bars"] == 2
    assert payload["policy"]["hold_bars"] == 288
    assert "[entry, exit)" in execution["funding"]
    assert "idle cash" in execution["cagr_clock"]


def test_manifest_binds_both_feature_sources_and_market() -> None:
    source = vtms.build_manifest()["source_contract"]
    assert source["spot_feature_sha256"] == (
        "d558239fa7085083aa002b7898b632df0774425719467709680ecb99718035a9"
    )
    assert source["perp_feature_sha256"] == (
        "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
    )
    assert source["market_sha256"] == (
        "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
    )


def test_manifest_hash_detects_mutation() -> None:
    payload = vtms.build_manifest()
    payload["policy"]["hold_bars"] = 144
    with pytest.raises(RuntimeError, match="hash mismatch"):
        vtms.validate_manifest(payload)


def test_write_once_refuses_changed_frozen_policy(tmp_path) -> None:
    path = tmp_path / "vtms.json"
    assert vtms.write_manifest_once(path, vtms.build_manifest()) == "created"
    assert vtms.write_manifest_once(path, vtms.build_manifest()) == "verified_existing"
    changed = json.loads(path.read_text())
    changed["selection_protocol"]["gates"]["2023_trades_min"] = 74
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = vtms.canonical_hash(core)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        vtms.write_manifest_once(path, vtms.build_manifest())
