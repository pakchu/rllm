from __future__ import annotations

import pytest

from training import evaluate_crrc_2023 as evaluate
from training import freeze_crrc_2023_evaluator as freeze


def test_manifest_freezes_all_dependencies_without_opening_outcomes() -> None:
    payload = freeze.build_manifest("a" * 40, primary_rows=156)
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["mutable_parameters"] == []
    assert payload["evaluation_config"] == evaluate.asdict(evaluate.CONFIG)
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False


def test_freeze_rejects_bad_commit_and_clock_count() -> None:
    with pytest.raises(ValueError, match="full Git hash"):
        freeze.build_manifest("short", primary_rows=156)
    payload = freeze.build_manifest("b" * 40, primary_rows=155)
    with pytest.raises(RuntimeError, match="clock count"):
        freeze.validate_manifest(payload)


def test_write_once_refuses_changed_payload(tmp_path) -> None:
    path = tmp_path / "freeze.json"
    first = freeze.build_manifest("a" * 40, primary_rows=156)
    assert freeze.write_once(path, first) == "created"
    assert freeze.write_once(path, first) == "verified_existing"
    second = freeze.build_manifest("b" * 40, primary_rows=156)
    with pytest.raises(RuntimeError, match="refusing"):
        freeze.write_once(path, second)
