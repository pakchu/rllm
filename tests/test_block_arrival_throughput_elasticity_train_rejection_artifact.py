from __future__ import annotations

import json

import pytest

from training import evaluate_block_arrival_throughput_elasticity_pre2024 as evaluate


TRAIN_SHA256 = "04150f70f838d092100a01b3fe6a07a6efb7e9f99b2ca05e164bb16ce14d15f4"
TRAIN_RESULT_HASH = (
    "486c8fd02ad016734aa6f6ab482b5ea87a0c07cc0633f6cf0fd17c41d190a093"
)


def test_bate288_train_rejection_is_hash_frozen_and_selection_stays_sealed() -> None:
    assert evaluate.sha256_file(evaluate.TRAIN_OUTPUT) == TRAIN_SHA256
    result = json.loads(evaluate.TRAIN_OUTPUT.read_text())
    evaluate.validate_result_hash(result)
    assert result["result_hash"] == TRAIN_RESULT_HASH
    assert result["protocol"]["opened_windows"] == ["train_2021_2022"]
    assert result["protocol"]["selection_2023_opened"] is False
    assert result["qualification"]["qualifies"] is False
    assert result["decision"] == "reject_before_selection"
    assert not evaluate.SELECTION_OUTPUT.exists()


def test_bate288_train_headlines_and_failure_reasons_are_stable() -> None:
    result = json.loads(evaluate.TRAIN_OUTPUT.read_text())
    primary = result["policies"]["primary"]
    base = primary["base_6bp"]
    assert base["absolute_return_pct"] == pytest.approx(29.98860509352237)
    assert base["cagr_pct"] == pytest.approx(14.022786556303624)
    assert base["strict_mdd_pct"] == pytest.approx(81.96846397236325)
    assert base["cagr_to_strict_mdd"] == pytest.approx(0.17107538529734545)
    assert base["trade_count"] == 644
    assert primary["stress_10bp"]["absolute_return_pct"] == pytest.approx(
        -22.37782504114133
    )
    assert primary["splits_base_6bp"]["train_2022"][
        "absolute_return_pct"
    ] == pytest.approx(-66.58851758959098)
    assert primary["side_contributions_base_6bp"]["HIGH_long"][
        "absolute_return_pct"
    ] == pytest.approx(-6.419800911358631)
    failures = result["qualification"]["failures"]
    assert "train: strict MDD above 15%" in failures
    assert "train_2022: non-positive absolute return" in failures


def test_failed_train_cannot_unlock_selection() -> None:
    freeze = evaluate.verify_evaluation_freeze()
    controls, _ = evaluate.verify_support_and_control_clocks()
    with pytest.raises(PermissionError, match="remains sealed because train failed"):
        evaluate._verify_passing_train_result(
            evaluate.EvaluationConfig(), freeze, controls
        )
