from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from training import preregister_aggregate_fill_compression_sweep as afcs


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = afcs.build_manifest()
    afcs.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(afcs.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]


def test_policy_uses_distinct_fill_compression_axis_and_safe_execution() -> None:
    payload = afcs.build_manifest()
    features = payload["causal_feature_contract"]
    execution = payload["execution_contract"]
    assert "underlying_trades_per_agg_event" in features["compression"]
    assert payload["policy"]["execution_delay_bars"] == 2
    assert payload["policy"]["hold_bars"] == 144
    assert "[entry, exit)" in execution["funding"]
    assert "idle cash" in execution["cagr_clock"]


def test_manifest_binds_audited_sources() -> None:
    source = afcs.build_manifest()["source_contract"]
    assert source["feature_sha256"] == (
        "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
    )
    assert source["market_sha256"] == (
        "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
    )
    assert source["source_audit_sha256"] == (
        "5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6"
    )


def test_manifest_hash_detects_mutation() -> None:
    payload = afcs.build_manifest()
    payload["policy"]["hold_bars"] = 145
    with pytest.raises(RuntimeError, match="hash mismatch"):
        afcs.validate_manifest(payload)


def test_write_once_refuses_changed_frozen_policy(tmp_path) -> None:
    path = tmp_path / "afcs.json"
    assert afcs.write_manifest_once(path, afcs.build_manifest()) == "created"
    assert afcs.write_manifest_once(path, afcs.build_manifest()) == "verified_existing"
    changed = json.loads(path.read_text())
    changed["selection_protocol"]["gates"]["2023_trades_min"] = 59
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = afcs.canonical_hash(core)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        afcs.write_manifest_once(path, afcs.build_manifest())
