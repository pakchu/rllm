from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from training import preregister_liquidity_vacuum_replenishment as lvrt


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = lvrt.build_manifest()
    lvrt.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(lvrt.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == [
        "2024",
        "2025",
        "2026_ytd",
    ]


def test_manifest_binds_audited_source_hashes() -> None:
    source = lvrt.build_manifest()["source_contract"]
    assert source["feature_sha256"] == (
        "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
    )
    assert source["market_sha256"] == (
        "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
    )
    assert source["source_audit_sha256"] == (
        "5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6"
    )


def test_time_shift_and_permutation_are_unambiguous_rejection_controls() -> None:
    controls = lvrt.build_manifest()["falsification_controls"]
    assert controls["sign_permuted_confirmation"].endswith("rejection control")
    assert "reject" in controls["placebo_rule"]
    assert "time-shift" in controls["placebo_rule"]
    assert "sign-permutation" in controls["placebo_rule"]


def test_manifest_hash_detects_policy_mutation() -> None:
    payload = lvrt.build_manifest()
    payload["policy"]["hold_bars"] = 13
    with pytest.raises(RuntimeError, match="hash mismatch"):
        lvrt.validate_manifest(payload)


def test_manifest_rejects_opened_outcomes_even_with_rehashed_payload() -> None:
    payload = lvrt.build_manifest()
    payload["outcomes_opened"] = True
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    payload["manifest_hash"] = lvrt.canonical_hash(core)
    with pytest.raises(RuntimeError, match="cannot open outcomes"):
        lvrt.validate_manifest(payload)


def test_write_once_verifies_same_policy_and_refuses_changed_policy(tmp_path) -> None:
    path = tmp_path / "lvrt.json"
    payload = lvrt.build_manifest()
    assert lvrt.write_manifest_once(path, payload) == "created"
    assert lvrt.write_manifest_once(path, lvrt.build_manifest()) == "verified_existing"

    changed = json.loads(path.read_text())
    changed["selection_protocol"]["gates"]["2023_trades_min"] = 79
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = lvrt.canonical_hash(core)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        lvrt.write_manifest_once(path, lvrt.build_manifest())
