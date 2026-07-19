from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_inverse_collateral_liquidation_absorption as evaluator


RESULT = Path("results/inverse_collateral_liquidation_absorption_train_2026-07-19.json")
RESULT_SHA256 = "093f137ff1447bac3c4c4d24b2e3c48e046390cde686e78f52d453b022ded00f"
RESULT_HASH = "5931d8e9df3026be8cb858806d540d00bbb93eff853865acb0fa4694dc4d5d44"


def test_icla_train_failed_and_later_outcomes_remain_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    result = json.loads(RESULT.read_text())

    assert result["result_hash"] == RESULT_HASH
    assert evaluator._stable_hash(result, "result_hash") == RESULT_HASH
    assert result["stage"] == "train"
    assert result["protocol"]["opened_windows"] == ["train"]
    assert result["protocol"]["sealed_windows"] == ["test", "eval"]
    assert result["protocol"]["loaded_market_windows"] == ["train"]
    assert result["protocol"]["loaded_funding_windows"] == ["train"]
    assert result["protocol"]["parameters_mutated_after_freeze"] is False

    base = result["base"]["metrics"]
    assert base["absolute_return_pct"] == pytest.approx(-7.613295874903869)
    assert base["cagr_pct"] == pytest.approx(-22.758754034986428)
    assert base["strict_mdd_pct"] == pytest.approx(9.104609914336443)
    assert base["cagr_to_strict_mdd"] == pytest.approx(-2.4996956760497433)
    assert base["executable_trades"] == 30
    assert base["long_trades"] == 15
    assert base["short_trades"] == 15
    assert base["mean_gross_trade_bps"] == pytest.approx(-14.109826785020847)
    assert base["mean_net_trade_bps"] == pytest.approx(-26.20315292193435)

    assert result["stress"]["metrics"]["absolute_return_pct"] == pytest.approx(
        -9.811185361171603
    )
    assert result["bootstrap"]["one_sided_p_value"] == pytest.approx(0.9857002859942802)
    assert result["promotion"]["passes"] is False
    assert result["promotion"]["checks"]["absolute_return_positive"] is False
    assert result["promotion"]["checks"]["stress_absolute_return_positive"] is False
    assert all(
        control["complete_gate"]["passes"] is False
        for control in result["controls"].values()
    )

    assert not evaluator.RESULT_PATHS["test"].exists()
    assert not evaluator.RESULT_PATHS["eval"].exists()
