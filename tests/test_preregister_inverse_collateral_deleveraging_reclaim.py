from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from training import preregister_inverse_collateral_deleveraging_reclaim as prereg


def test_manifest_is_deterministic_singleton_and_outcome_blind() -> None:
    first = prereg.build_manifest()
    second = prereg.build_manifest()
    assert first == second
    assert first["outcomes_opened"] is False
    assert first["selection_protocol"]["candidate_count"] == 1
    assert first["causal_feature_contract"]["price_signal_columns"] == []
    assert first["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
    prereg.validate_manifest(first)


def test_policy_freezes_causal_delay_units_and_only_support_grid() -> None:
    policy = asdict(prereg.Policy())
    assert policy["oi_change_bars"] == 12
    assert policy["taker_smoothing_bars"] == 3
    assert policy["confirmation_window_bars"] == 12
    assert policy["execution_delay_bars"] == 2
    assert policy["hold_bars"] == 144
    assert policy["purge_quantiles"] == (0.80, 0.85, 0.90, 0.925, 0.95)
    manifest = prereg.build_manifest()
    assert manifest["policy"]["purge_quantiles"] == [0.80, 0.85, 0.90, 0.925, 0.95]
    assert manifest["support_calibration"]["vary_only"].startswith(
        "relative purge quantile"
    )
    assert "2021-2022" in manifest["support_calibration"]["selection_rule"]
    assert (
        "cannot select a fallback" in manifest["support_calibration"]["selection_rule"]
    )
    assert manifest["selection_protocol"]["stage2_support_cannot_reselect_q"] is True
    units = manifest["causal_feature_contract"]["unit_safe_open_interest"]
    assert "um_sum_open_interest_value" in units["um"]
    assert "cm_sum_open_interest" in units["cm"]


def test_validate_manifest_rejects_drift_and_opened_outcomes() -> None:
    opened = prereg.build_manifest()
    opened["outcomes_opened"] = True
    opened["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in opened.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="outcomes opened"):
        prereg.validate_manifest(opened, verify_sources=False)

    drifted = prereg.build_manifest()
    drifted["policy"]["hold_bars"] = 12
    drifted["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in drifted.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy differs"):
        prereg.validate_manifest(drifted, verify_sources=False)


def test_frozen_source_hashes_match() -> None:
    source = prereg.build_manifest()["source_contract"]
    for key, hash_key in (
        ("metrics", "metrics_sha256"),
        ("metrics_manifest", "metrics_manifest_sha256"),
        ("market", "market_sha256"),
        ("market_manifest", "market_manifest_sha256"),
        ("funding", "funding_sha256"),
        ("funding_manifest", "funding_manifest_sha256"),
    ):
        assert (
            hashlib.sha256(Path(source[key]).read_bytes()).hexdigest()
            == source[hash_key]
        )


def test_written_preregistration_artifact_replays() -> None:
    payload = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    assert payload == prereg.build_manifest()
    prereg.validate_manifest(payload)
