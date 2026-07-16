from __future__ import annotations

from dataclasses import asdict

import pytest

from training import preregister_blockspace_fee_confirmation as bfc


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = bfc.build_manifest()
    bfc.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(bfc.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]


def test_signal_is_price_independent_and_exchange_tags_are_excluded() -> None:
    payload = bfc.build_manifest()
    assert payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"] == []
    assert "FlowInEx" in payload["source_contract"]["excluded_for_leakage_risk"]


def test_source_hashes_are_frozen() -> None:
    source = bfc.build_manifest()["source_contract"]
    assert source["blockspace_sha256"] == (
        "c94fd06ff695d673503a56064284cffbb36e6f1ac847bdc6b38819752a77985b"
    )
    assert source["blockspace_manifest_sha256"] == (
        "eb70f5cd38d0895b9c04ca142ce15645696769f10866072b8f8ef64ec7a49cf1"
    )
    assert source["database_snapshot_is_point_in_time"] is False


def test_execution_and_hold_are_fixed() -> None:
    payload = bfc.build_manifest()
    assert payload["causal_feature_contract"]["direction"] == "long only"
    assert payload["policy"]["hold_bars"] == 3 * 288
    assert payload["execution_contract"]["nonoverlap"] is True
    assert payload["policy"]["entry_delay_bars"] == 1


def test_stale_backfill_cannot_signal() -> None:
    payload = bfc.build_manifest()
    assert payload["policy"]["maximum_source_lag_days"] == 3.0
    assert payload["support_freeze_before_returns"][
        "stale_backfill_rows_may_seed_reference_but_may_not_signal"
    ] is True


def test_direction_flip_is_diagnostic_only() -> None:
    control = bfc.build_manifest()["controls"]["direction_flip"]
    assert "diagnostic-only" in control
    assert "never" in control


def test_manifest_hash_detects_mutation() -> None:
    payload = bfc.build_manifest()
    payload["policy"]["hold_bars"] += 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        bfc.validate_manifest(payload)


def test_write_once_refuses_different_frozen_policy(tmp_path) -> None:
    path = tmp_path / "bfc.json"
    assert bfc.write_once(path, bfc.build_manifest()) == "created"
    assert bfc.write_once(path, bfc.build_manifest()) == "verified_existing"
    changed = bfc.build_manifest()
    changed["policy"]["composite_min"] = 1.6
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = bfc.canonical_hash(core)
    path.write_text(__import__("json").dumps(changed))
    with pytest.raises(RuntimeError, match="policy differs from code"):
        bfc.write_once(path, bfc.build_manifest())
