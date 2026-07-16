from __future__ import annotations

import json

import pytest

from training import freeze_network_weak_signal_ensemble_v2_evaluator as freeze


def test_manifest_declares_zero_outcome_access() -> None:
    payload = freeze.build_manifest("a" * 40, feature_clock_rows=208)
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["labels_constructed_during_freeze"] is False
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []


def test_manifest_tampering_is_detected() -> None:
    payload = freeze.build_manifest("b" * 40, feature_clock_rows=208)
    payload["market_rows_parsed_during_freeze"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze.validate_manifest(payload)


def test_write_once_refuses_a_different_freeze(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    payload = freeze.build_manifest("c" * 40, feature_clock_rows=208)
    assert freeze.write_once(output, payload) == "created"
    frozen = json.loads(output.read_text())
    freeze.validate_manifest(frozen)
    changed = freeze.build_manifest("d" * 40, feature_clock_rows=208)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        freeze.write_once(output, changed)
