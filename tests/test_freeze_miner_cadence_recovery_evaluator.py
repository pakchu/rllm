from __future__ import annotations

import json

import pytest

from training import evaluate_miner_cadence_recovery_pre2024 as evaluate
from training import freeze_miner_cadence_recovery_evaluator as freeze


HASHES = {name: name * 2 for name in evaluate.POLICY_NAMES}
COUNTS = {name: 10 for name in evaluate.POLICY_NAMES}


def test_manifest_declares_zero_outcome_access() -> None:
    payload = freeze.build_manifest(
        "a" * 40,
        control_clock_hashes=HASHES,
        control_clock_counts=COUNTS,
    )
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["labels_constructed_during_freeze"] is False
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []
    assert payload["opened_windows"] == []


def test_manifest_tampering_is_detected() -> None:
    payload = freeze.build_manifest(
        "b" * 40,
        control_clock_hashes=HASHES,
        control_clock_counts=COUNTS,
    )
    payload["market_rows_parsed_during_freeze"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze.validate_manifest(payload)


def test_write_once_refuses_a_different_freeze(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    payload = freeze.build_manifest(
        "c" * 40,
        control_clock_hashes=HASHES,
        control_clock_counts=COUNTS,
    )
    assert freeze.write_once(output, payload) == "created"
    frozen = json.loads(output.read_text())
    freeze.validate_manifest(frozen)
    changed = freeze.build_manifest(
        "d" * 40,
        control_clock_hashes=HASHES,
        control_clock_counts=COUNTS,
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        freeze.write_once(output, changed)


def test_manifest_requires_every_control_clock() -> None:
    hashes = dict(HASHES)
    hashes.pop("primary")
    with pytest.raises(ValueError, match="lacks a control-clock hash"):
        freeze.build_manifest(
            "e" * 40,
            control_clock_hashes=hashes,
            control_clock_counts=COUNTS,
        )
