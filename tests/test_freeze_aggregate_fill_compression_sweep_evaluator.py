from __future__ import annotations

import json

import pytest

from training import evaluate_aggregate_fill_compression_sweep as evaluate
from training import freeze_aggregate_fill_compression_sweep_evaluator as freeze


def test_build_manifest_is_outcome_blind_and_binds_all_clocks() -> None:
    payload = evaluate.build_freeze_manifest("0" * 40)
    freeze.validate_manifest(payload)
    assert payload["opened_windows"] == []
    assert payload["mutable_parameters"] == []
    assert payload["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert payload["funding_settlement_marks_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert set(payload["control_schedules"]) == set(evaluate.support.POLICY_NAMES)


def test_write_once_is_idempotent_and_refuses_mutation(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    payload = evaluate.build_freeze_manifest("0" * 40)
    assert freeze.write_once(output, payload) == "created"
    assert freeze.write_once(output, payload) == "verified_existing"
    changed = json.loads(output.read_text())
    changed["opened_windows"] = ["stage1"]
    output.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        freeze.write_once(output, payload)


def test_current_clean_source_commit_reproduces_evaluator() -> None:
    commit = freeze.current_clean_source_commit()
    assert len(commit) == 40
