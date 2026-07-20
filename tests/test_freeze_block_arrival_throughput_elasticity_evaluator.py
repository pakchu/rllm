from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import evaluate_block_arrival_throughput_elasticity_pre2024 as evaluate
from training import freeze_block_arrival_throughput_elasticity_evaluator as freeze


HASHES = {name: name * 2 for name in evaluate.POLICY_NAMES}
COUNTS = {name: 10 for name in evaluate.POLICY_NAMES}
BOUNDARIES = {
    "market": {
        "value_rows_parsed": 0,
        "window_value_row_counts": {"train": 100, "selection": 50},
    },
    "funding": {
        "value_rows_parsed": 0,
        "window_value_row_counts": {"train": 10, "selection": 5},
    },
}


def _manifest(commit: str = "a" * 40) -> dict[str, object]:
    return freeze.build_manifest(
        commit,
        control_clock_hashes=HASHES,
        control_clock_counts=COUNTS,
        outcome_boundaries=BOUNDARIES,
    )


def test_manifest_declares_zero_outcome_access() -> None:
    payload = _manifest()
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["labels_constructed_during_freeze"] is False
    assert payload["market_value_rows_parsed_during_freeze"] == 0
    assert payload["funding_value_rows_parsed_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []
    assert payload["opened_windows"] == []


def test_manifest_tampering_is_detected() -> None:
    payload = _manifest()
    payload["market_value_rows_parsed_during_freeze"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze.validate_manifest(payload)


def test_write_once_refuses_a_different_freeze(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    payload = _manifest("b" * 40)
    assert freeze.write_once(output, payload) == "created"
    frozen = json.loads(output.read_text())
    freeze.validate_manifest(frozen)
    changed = _manifest("c" * 40)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        freeze.write_once(output, changed)


def test_manifest_requires_every_control_clock() -> None:
    hashes = dict(HASHES)
    hashes.pop("primary")
    with pytest.raises(ValueError, match="lacks a control-clock hash"):
        freeze.build_manifest(
            "d" * 40,
            control_clock_hashes=hashes,
            control_clock_counts=COUNTS,
            outcome_boundaries=BOUNDARIES,
        )


def test_manifest_rejects_boundary_scan_that_parsed_values() -> None:
    boundaries = json.loads(json.dumps(BOUNDARIES))
    boundaries["funding"]["value_rows_parsed"] = 1
    with pytest.raises(ValueError, match="parsed outcome values"):
        freeze.build_manifest(
            "e" * 40,
            control_clock_hashes=HASHES,
            control_clock_counts=COUNTS,
            outcome_boundaries=boundaries,
        )
