from __future__ import annotations

from dataclasses import asdict

import pytest

from training import preregister_network_topology_broadening as ntb


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = ntb.build_manifest()
    ntb.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(ntb.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]


def test_signal_clock_has_no_market_or_derivative_feature() -> None:
    payload = ntb.build_manifest()
    assert payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"] == []
    assert any(
        "price momentum" in exclusion
        for exclusion in payload["novelty_boundary"]["not"]
    )


def test_source_hashes_and_availability_are_explicit() -> None:
    source = ntb.build_manifest()["source_contract"]
    assert source["network_sha256"] == (
        "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
    )
    assert source["network_manifest_sha256"] == (
        "66b185769800c4732cf748b40ca9cb48c5eee239abf0425ff193c0688111c372"
    )
    assert source["database_snapshot_is_point_in_time"] is False
    assert "AssetEODCompletionTime" in source["availability_column"]


def test_policy_is_long_only_and_fixed_for_seven_days() -> None:
    payload = ntb.build_manifest()
    assert payload["causal_feature_contract"]["direction"] == "long only"
    assert payload["policy"]["hold_bars"] == 7 * 288
    assert payload["execution_contract"]["nonoverlap"] is True


def test_stale_backfill_cannot_emit_signal() -> None:
    payload = ntb.build_manifest()
    assert payload["policy"]["maximum_source_lag_days"] == 3.0
    assert payload["support_freeze_before_returns"][
        "stale_backfill_rows_may_seed_reference_but_may_not_signal"
    ] is True


def test_direction_flip_is_diagnostic_only() -> None:
    control = ntb.build_manifest()["controls"]["direction_flip"]
    assert "diagnostic-only" in control
    assert "never" in control


def test_manifest_hash_detects_mutation() -> None:
    payload = ntb.build_manifest()
    payload["policy"]["hold_bars"] += 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ntb.validate_manifest(payload)


def test_write_once_refuses_different_frozen_policy(tmp_path) -> None:
    path = tmp_path / "ntb.json"
    assert ntb.write_once(path, ntb.build_manifest()) == "created"
    assert ntb.write_once(path, ntb.build_manifest()) == "verified_existing"
    changed = ntb.build_manifest()
    changed["policy"]["composite_min"] = 1.6
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = ntb.canonical_hash(core)
    path.write_text(__import__("json").dumps(changed))
    with pytest.raises(RuntimeError, match="policy differs from code"):
        ntb.write_once(path, ntb.build_manifest())
