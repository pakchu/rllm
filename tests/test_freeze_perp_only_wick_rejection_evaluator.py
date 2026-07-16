from __future__ import annotations

import pytest

from training import freeze_perp_only_wick_rejection_evaluator as freeze


def _clocks() -> tuple[dict[str, str], dict[str, int]]:
    hashes = {name: "0" * 64 for name in freeze.evaluate.POLICY_NAMES}
    hashes["primary"] = freeze.evaluate.EVENT_CLOCK_SHA256
    rows = {name: 1 for name in freeze.evaluate.POLICY_NAMES}
    return hashes, rows


def test_freeze_manifest_is_outcome_blind_and_immutable() -> None:
    hashes, rows = _clocks()
    payload = freeze.build_manifest(
        "a" * 40,
        policy_clock_sha256=hashes,
        policy_clock_rows=rows,
    )
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["mutable_parameters"] == []
    assert payload["signal_feature_ohlc_loaded_during_freeze"] is True
    assert payload["post_signal_returns_computed_during_freeze"] is False
    assert payload["funding_loaded_during_freeze"] is False
    assert payload["execution_simulation_run_during_freeze"] is False


def test_freeze_hash_detects_mutation() -> None:
    hashes, rows = _clocks()
    payload = freeze.build_manifest(
        "b" * 40,
        policy_clock_sha256=hashes,
        policy_clock_rows=rows,
    )
    payload["opened_windows"].append("train")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze.validate_manifest(payload)


def test_freeze_requires_full_commit_hash() -> None:
    hashes, rows = _clocks()
    with pytest.raises(ValueError, match="full Git hash"):
        freeze.build_manifest(
            "abc123",
            policy_clock_sha256=hashes,
            policy_clock_rows=rows,
        )


def test_repository_freeze_verifies_before_outcomes() -> None:
    payload = freeze.evaluate.verify_evaluation_freeze()
    assert payload["outcomes_opened"] is False
    assert payload["post_signal_returns_computed_during_freeze"] is False
    assert payload["policy_clock_rows"]["primary"] == 637
