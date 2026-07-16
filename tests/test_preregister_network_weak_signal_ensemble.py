from __future__ import annotations

from dataclasses import asdict

import pytest

from training import preregister_network_weak_signal_ensemble as nwe


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = nwe.build_manifest()
    nwe.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["policy"] == asdict(nwe.Policy())
    assert payload["feature_columns"] == list(nwe.FEATURE_COLUMNS)
    assert payload["selection_protocol"]["candidate_count"] == 1


def test_support_loads_no_market_features_or_labels() -> None:
    payload = nwe.build_manifest()
    assert payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"] == []
    assert payload["support_freeze_before_labels"]["market_or_return_rows_loaded"] == 0


def test_model_removes_unconditional_drift() -> None:
    model = nwe.build_manifest()["online_model_contract"]
    assert "do not add it back" in model["target_centering"]
    assert "no intercept" in model["estimator"]


def test_training_labels_are_strictly_available() -> None:
    rule = nwe.build_manifest()["online_model_contract"]["label_availability"]
    assert "label exit" in rule
    assert "<= current weekly decision" in rule


def test_source_hashes_are_frozen() -> None:
    source = nwe.build_manifest()["source_contract"]
    assert source["blockspace_sha256"] == (
        "c94fd06ff695d673503a56064284cffbb36e6f1ac847bdc6b38819752a77985b"
    )
    assert source["network_sha256"] == (
        "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
    )
    assert "exchange-tag" in source["excluded_for_leakage_risk"]


def test_execution_is_fixed_weekly() -> None:
    policy = nwe.Policy()
    assert policy.decision_weekday == 0
    assert policy.decision_hour_utc == 12
    assert policy.hold_bars == 7 * 288
    assert policy.entry_delay_bars == 1


def test_manifest_hash_detects_mutation() -> None:
    payload = nwe.build_manifest()
    payload["policy"]["ridge_alpha"] = 9.0
    with pytest.raises(RuntimeError, match="hash mismatch"):
        nwe.validate_manifest(payload)


def test_write_once_refuses_different_frozen_policy(tmp_path) -> None:
    path = tmp_path / "nwe.json"
    assert nwe.write_once(path, nwe.build_manifest()) == "created"
    assert nwe.write_once(path, nwe.build_manifest()) == "verified_existing"
    changed = nwe.build_manifest()
    changed["policy"]["maximum_train_samples"] = 105
    core = {
        key: value
        for key, value in changed.items()
        if key not in {"manifest_hash", "created_at"}
    }
    changed["manifest_hash"] = nwe.canonical_hash(core)
    path.write_text(__import__("json").dumps(changed))
    with pytest.raises(RuntimeError, match="policy differs from code"):
        nwe.write_once(path, nwe.build_manifest())
