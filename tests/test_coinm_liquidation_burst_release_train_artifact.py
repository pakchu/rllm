from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_coinm_liquidation_burst_release as evaluator


TRAIN_RESULT = Path(
    "results/coinm_liquidation_burst_release_train_2026-07-19.json"
)
EXPECTED_SHA256 = "5b52d00cca61e2b4f41c613908d9a55dbc4666a9e58ce453d8fa72e5fa2774d3"


def test_clbr_train_failure_is_exact_and_later_windows_remain_sealed() -> None:
    assert hashlib.sha256(TRAIN_RESULT.read_bytes()).hexdigest() == EXPECTED_SHA256
    result = json.loads(TRAIN_RESULT.read_text())
    assert result["result_hash"] == (
        "2a6af9a53bba1e401ab9abc7b8872b0a5def9e46f95f585b794878723faddd63"
    )
    assert result["stage"] == "train"
    assert result["protocol"] == {
        "opened_windows": ["train"],
        "sealed_windows": ["test", "eval"],
        "loaded_market_windows": ["train"],
        "loaded_funding_windows": ["train"],
        "loaded_clock_windows": ["train"],
        "parameters_mutated_after_freeze": False,
    }
    metrics = result["base"]["metrics"]
    assert metrics["absolute_return_pct"] == pytest.approx(-6.0660970322122365)
    assert metrics["cagr_pct"] == pytest.approx(-18.459908734966213)
    assert metrics["strict_mdd_pct"] == pytest.approx(7.340401898754847)
    assert metrics["cagr_to_strict_mdd"] == pytest.approx(-2.514836243244061)
    assert metrics["executable_trades"] == 38
    assert metrics["long_trades"] == 28
    assert metrics["short_trades"] == 10
    assert result["stress"]["metrics"]["absolute_return_pct"] == pytest.approx(
        -10.261011088783933
    )
    assert result["bootstrap"]["one_sided_p_value"] == 1.0
    assert result["promotion"]["passes"] is False
    assert result["promotion"]["checks"]["absolute_return_positive"] is False
    assert result["promotion"]["checks"]["cagr_to_strict_mdd"] is False
    assert not evaluator.RESULT_PATHS["test"].exists()
    assert not evaluator.RESULT_PATHS["eval"].exists()


def test_failed_train_cannot_unlock_test() -> None:
    freeze = evaluator.verify_evaluator_freeze()
    with pytest.raises(ValueError, match="failed; later windows remain sealed"):
        evaluator._verify_prior_result("train", evaluator.EvaluationConfig(), freeze)
