from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RESULT = Path(
    "results/causal_adaptive_relational_baseline_selection_2026-07-14.json"
)
RESULT_SHA256 = "b17ef30fd97bc8054a49e42c84d406439c547b97fbd8fb94f0baf59625c55a75"


def _result() -> dict[str, object]:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    return json.loads(RESULT.read_text())


def test_carta_baseline_rejection_is_frozen_and_oos_remains_sealed() -> None:
    result = _result()
    assert result["selection"] == {
        "selected_policy": None,
        "rejected": True,
        "gemma_stage_allowed": False,
        "reason": "no cheap causal CARTA baseline passed the frozen learnability gate",
    }
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]


def test_relational_ridge_failed_and_executed_direction_collapsed() -> None:
    result = _result()
    ridge = result["policies"]["relational_ridge"]
    metrics = ridge["windows"]["select2023"]
    assert metrics["absolute_return_pct"] == pytest.approx(-0.7448757208051493)
    assert metrics["cagr_to_strict_mdd"] == pytest.approx(-0.3644910971078076)
    assert metrics["trade_count"] == 31
    assert metrics["long_count"] == 28
    assert metrics["short_count"] == 3
    assert ridge["qualification"]["qualifies"] is False


def test_naive_bayes_lost_in_both_2023_halves() -> None:
    result = _result()
    windows = result["policies"]["naive_bayes"]["windows"]
    assert windows["select2023_h1"]["absolute_return_pct"] < 0.0
    assert windows["select2023_h2"]["absolute_return_pct"] < 0.0
