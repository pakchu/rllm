from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from training import preregister_network_weak_signal_ensemble_v2 as prereg


def test_manifest_is_singleton_price_free_and_sealed() -> None:
    payload = prereg.build_manifest()
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(prereg.Policy())
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == ["2024", "2025", "2026_ytd"]
    assert payload["causal_feature_contract"][
        "price_or_derivative_feature_columns_loaded"
    ] == []


def test_warmup_change_is_explicit_and_predecessor_outcomes_remain_sealed() -> None:
    payload = prereg.build_manifest()
    assert payload["policy"]["prediction_start"] == "2021-06-07"
    assert payload["policy"]["minimum_train_samples"] == 52
    assert payload["research_history_boundary"]["nwe7_support_result_opened"] is True
    assert payload["research_history_boundary"]["nwe7_return_labels_or_pnl_opened"] is False
    assert payload["support_freeze_before_labels"][
        "initial_fully_available_training_samples_min"
    ] == 52


def test_tampering_is_detected() -> None:
    payload = prereg.build_manifest()
    payload["policy"]["ridge_alpha"] = 1.0
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.validate_manifest(payload)


def test_write_once_refuses_manifest_change(tmp_path) -> None:
    output = tmp_path / "prereg.json"
    payload = prereg.build_manifest()
    assert prereg.write_once(output, payload) == "created"
    frozen = json.loads(output.read_text())
    prereg.validate_manifest(frozen)
    changed = prereg.build_manifest()
    changed["policy"]["prediction_start"] = "2021-06-14"
    changed_core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = prereg.canonical_hash(changed_core)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        prereg.write_once(output, changed)
