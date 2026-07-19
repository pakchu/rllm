from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_eth_btc_liquidation_relay as evaluator


RESULT = Path("results/eth_btc_liquidation_relay_train_2026-07-19.json")
RESULT_SHA256 = "1829a482bf0f99aea35814a6050ac3f6ad844bf8ab4aa7405332ddcbe29403fa"
RESULT_HASH = "7d42383aa2fd87bc8622870c5d7a88f01a1a972b67fb366ac613eb024d429f3d"


def test_eblr_train_failed_and_test_eval_remain_sealed() -> None:
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
    assert base["absolute_return_pct"] == pytest.approx(-1.814610348679957)
    assert base["cagr_pct"] == pytest.approx(-5.7971394208709555)
    assert base["strict_mdd_pct"] == pytest.approx(3.347638064347442)
    assert base["cagr_to_strict_mdd"] == pytest.approx(-1.7317103311169924)
    assert base["executable_trades"] == 21
    assert base["long_trades"] == 8
    assert base["short_trades"] == 13
    assert base["mean_gross_trade_bps"] == pytest.approx(3.3441472866246933)
    assert base["mean_net_trade_bps"] == pytest.approx(-8.651283197037033)
    assert result["stress"]["metrics"]["absolute_return_pct"] == pytest.approx(
        -3.451812488850925
    )
    assert result["bootstrap"]["one_sided_p_value"] == pytest.approx(
        0.874642507149857
    )
    assert result["promotion"]["passes"] is False
    assert result["promotion"]["checks"]["absolute_return_positive"] is False
    assert result["promotion"]["checks"]["stress_absolute_return_positive"] is False
    assert result["promotion"]["checks"]["bootstrap_p_value"] is False
    assert all(
        control["complete_gate"]["passes"] is False
        for control in result["controls"].values()
    )
    placebo = result["controls"]["future_eth_placebo"]
    assert placebo["base"]["metrics"]["absolute_return_pct"] == pytest.approx(
        2.6492018916863813
    )
    assert placebo["base"]["metrics"]["cagr_to_strict_mdd"] == pytest.approx(
        6.037606822084234
    )
    assert placebo["bootstrap"]["one_sided_p_value"] == pytest.approx(
        0.14265714685706285
    )

    assert not evaluator.RESULT_PATHS["test"].exists()
    assert not evaluator.RESULT_PATHS["eval"].exists()
