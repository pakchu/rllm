from __future__ import annotations

from dataclasses import asdict

import pytest

from training import preregister_perp_only_wick_rejection as powr


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = powr.build_manifest()
    powr.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(powr.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]


def test_source_hashes_and_backfill_boundary_are_explicit() -> None:
    source = powr.build_manifest()["source_contract"]
    assert source["perp_sha256"] == (
        "0b55bb0c3b845a90da738e746c769b19c1de4ac230ca8f1fccb6c361c4a9a41f"
    )
    assert source["spot_sha256"] == (
        "bc6e0fd6b773ab6458a5de88fb9589161d1adf4ac1d0e7024f252515909f4a54"
    )
    assert source["database_snapshot_is_point_in_time"] is False


def test_execution_does_not_filter_on_future_hold_spot_availability() -> None:
    payload = powr.build_manifest()
    support = payload["support_freeze_before_returns"]
    assert support["signal_and_latency_joint_buckets_complete"] is True
    assert support["future_hold_spot_availability_used_to_filter"] is False
    assert payload["policy"]["entry_delay_bars"] == 3
    assert "t+15m" in payload["execution_contract"]["entry"]


def test_direction_flip_is_unambiguously_diagnostic_only() -> None:
    direction_flip = powr.build_manifest()["controls"]["direction_flip"]
    assert "diagnostic-only" in direction_flip
    assert "never" in direction_flip


def test_delayed_entry_control_is_one_bar_after_primary() -> None:
    payload = powr.build_manifest()
    assert payload["policy"]["entry_delay_bars"] == 3
    delayed = payload["controls"]["one_bar_delayed_entry"]
    assert "signal_position+4" in delayed
    assert "t+20m" in delayed


def test_manifest_hash_detects_mutation() -> None:
    payload = powr.build_manifest()
    payload["policy"]["hold_bars"] = 13
    with pytest.raises(RuntimeError, match="hash mismatch"):
        powr.validate_manifest(payload)


def test_write_once_refuses_different_frozen_policy(tmp_path) -> None:
    path = tmp_path / "powr.json"
    payload = powr.build_manifest()
    assert powr.write_once(path, payload) == "created"
    assert powr.write_once(path, powr.build_manifest()) == "verified_existing"
    changed = powr.build_manifest()
    changed["policy"]["minimum_perp_wick_bp"] = 7.0
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = powr.canonical_hash(core)
    path.write_text(__import__("json").dumps(changed))
    with pytest.raises(RuntimeError, match="policy differs from code"):
        powr.write_once(path, powr.build_manifest())
