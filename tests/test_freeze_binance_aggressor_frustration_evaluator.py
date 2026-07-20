from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from training import evaluate_binance_aggressor_frustration_pre2024 as evaluate
from training import freeze_binance_aggressor_frustration_evaluator as freeze


def _predictor_boundary() -> dict[str, object]:
    return {
        "official_market_value_rows_parsed": 0,
        "funding_value_rows_parsed": 0,
        "post_entry_outcome_rows_loaded": 0,
        "strategy_outcomes_calculated": False,
    }


def _outcome_boundaries() -> dict[str, object]:
    return {
        "market": {"value_rows_parsed": 0},
        "funding": {"value_rows_parsed": 0},
    }


def _manifest() -> dict[str, object]:
    names = evaluate.POLICY_NAMES
    return freeze.build_manifest(
        "a" * 40,
        control_clock_hashes={name: f"hash-{name}" for name in names},
        control_clock_counts={name: 1 for name in names},
        predictor_boundary=_predictor_boundary(),
        outcome_boundaries=_outcome_boundaries(),
    )


def test_manifest_freezes_all_policies_and_zero_outcome_reads() -> None:
    payload = _manifest()
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["market_value_rows_parsed_during_freeze"] == 0
    assert payload["funding_value_rows_parsed_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["evaluation_config"] == asdict(evaluate.EvaluationConfig())
    assert payload["policy_names"] == list(evaluate.POLICY_NAMES)
    assert payload["evaluation_test"] == str(evaluate.EVALUATION_TEST)
    assert payload["freeze_test"] == str(evaluate.FREEZE_TEST)
    assert payload["evaluation_test_sha256"] == evaluate.sha256_file(
        evaluate.EVALUATION_TEST
    )
    assert payload["freeze_test_sha256"] == evaluate.sha256_file(evaluate.FREEZE_TEST)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_market_value_rows_parsed", 1),
        ("funding_value_rows_parsed", 1),
        ("post_entry_outcome_rows_loaded", 1),
        ("strategy_outcomes_calculated", True),
    ],
)
def test_manifest_rejects_predictor_boundary_leak(field: str, value: object) -> None:
    predictor = _predictor_boundary()
    predictor[field] = value
    with pytest.raises(ValueError):
        freeze.build_manifest(
            "a" * 40,
            control_clock_hashes={name: "hash" for name in evaluate.POLICY_NAMES},
            control_clock_counts={name: 1 for name in evaluate.POLICY_NAMES},
            predictor_boundary=predictor,
            outcome_boundaries=_outcome_boundaries(),
        )


def test_manifest_rejects_outcome_boundary_value_parse() -> None:
    boundaries = _outcome_boundaries()
    boundaries["market"]["value_rows_parsed"] = 1
    with pytest.raises(ValueError, match="parsed outcome values"):
        freeze.build_manifest(
            "a" * 40,
            control_clock_hashes={name: "hash" for name in evaluate.POLICY_NAMES},
            control_clock_counts={name: 1 for name in evaluate.POLICY_NAMES},
            predictor_boundary=_predictor_boundary(),
            outcome_boundaries=boundaries,
        )


def test_write_once_is_idempotent_and_rejects_replacement(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    payload = _manifest()
    assert freeze.write_once(path, payload) == "created"
    assert freeze.write_once(path, payload) == "verified_existing"
    changed = dict(payload)
    changed["evaluation_source_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        freeze.write_once(path, changed)


def test_manifest_hash_tamper_fails() -> None:
    payload = _manifest()
    payload["opened_windows"] = ["train_2020_2022"]
    with pytest.raises(RuntimeError, match="hash mismatch"):
        freeze.validate_manifest(payload)
