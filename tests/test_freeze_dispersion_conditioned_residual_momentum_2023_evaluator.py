from __future__ import annotations

import json

import pytest

from training import freeze_dispersion_conditioned_residual_momentum_2023_evaluator as freeze


def test_manifest_declares_zero_outcome_access() -> None:
    payload = freeze.build_manifest("a" * 40, clock_rows=92)
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["source_prefix_contract"]["2024_rows_permitted"] == 0


def test_manifest_tampering_is_detected() -> None:
    payload = freeze.build_manifest("b" * 40, clock_rows=92)
    payload["market_rows_parsed_during_freeze"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze.validate_manifest(payload)


def test_write_once_refuses_a_different_freeze(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    payload = freeze.build_manifest("c" * 40, clock_rows=92)
    assert freeze.write_once(output, payload) == "created"
    frozen = json.loads(output.read_text())
    freeze.validate_manifest(frozen)
    changed = freeze.build_manifest("d" * 40, clock_rows=92)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        freeze.write_once(output, changed)


def test_current_evaluator_and_tests_are_reproducible_from_head() -> None:
    commit = freeze.current_clean_commit()
    assert len(commit) == 40
