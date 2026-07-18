from __future__ import annotations

import json

import pytest

from training import evaluate_coinm_roll_migration_pre2024_v2 as evaluator


def test_v2_static_support_advances_only_next_led_candidate() -> None:
    assert not evaluator.V1_ABORTED_SELECTION.exists()
    support = evaluator._verify_static_dependencies()
    passed = [
        item["candidate"]["name"]
        for item in support["candidates"]
        if item["passes_support"]
    ]
    assert passed == ["coinm_next_led_roll_migration_h60m"]
    assert support["protocol"]["candidate_return_statistics_opened_for_v2"] is False


def test_v2_evaluator_reuses_exact_frozen_v1_ledger() -> None:
    expected = evaluator.STATIC_INPUT_SHA256[
        "training/evaluate_coinm_roll_migration_pre2024.py"
    ]
    assert evaluator._sha256(
        "training/evaluate_coinm_roll_migration_pre2024.py"
    ) == expected


def test_v2_config_is_fully_immutable() -> None:
    with pytest.raises(ValueError, match="protocol parameters are frozen"):
        evaluator._require_canonical_config(
            evaluator.EvaluationConfig(cost_rate_per_side=0.0)
        )


def test_v2_source_identity_is_checked_without_opening_outcomes() -> None:
    cfg = evaluator.EvaluationConfig()
    assert evaluator._sha256(cfg.source_csv) == evaluator.SOURCE_SHA256
    assert evaluator._sha256(cfg.manifest_json) == evaluator.MANIFEST_SHA256
    manifest = json.loads(open(cfg.manifest_json).read())
    assert manifest["last_signal_bar"] == "2023-12-31 23:50:00"
    assert manifest["protocol"]["post2023_opened"] is False
