from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from training import preregister_cross_collateral_liquidity_vacuum as clv


def _cfg() -> clv.Config:
    return replace(
        clv.Config(),
        response_bars=2,
        robust_baseline_bars=4,
        robust_min_periods=2,
        score_baseline_bars=4,
        score_min_periods=2,
        score_quantile=0.5,
        hold_bars=2,
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_half=0,
        minimum_nonoverlap_per_quarter=0,
        minimum_side_share=0.0,
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=9, freq="5min"),
            "clean": [True] * 9,
            "flow_aligned": [True] * 9,
            "direction": [1] * 8 + [-1],
            "response_return_z": [1.0] * 4 + [10.0, 1.0, 1.0, 1.0, 10.0],
            "level_response_z": [1.0] * 4 + [10.0, 1.0, 1.0, 1.0, -10.0],
            "shape_response_z": [1.0] * 4 + [10.0, 1.0, 1.0, 1.0, -10.0],
        }
    )


def test_clv_retains_vacuum_and_drops_refill() -> None:
    signal = clv.classify_vacuum(_features(), _cfg())
    assert signal.loc[4, "candidate"]
    assert signal.loc[4, "branch"] == "vacuum"
    assert signal.loc[4, "side"] == 1
    assert not signal.loc[8, "candidate"]
    assert signal.loc[8, "branch"] == "none"
    assert signal.loc[8, "side"] == 0


def test_vacuum_follows_negative_shock() -> None:
    features = _features()
    features.loc[4, "direction"] = -1
    signal = clv.classify_vacuum(features, _cfg())
    assert signal.loc[4, "candidate"]
    assert signal.loc[4, "side"] == -1


def test_support_stopping_rule_is_inherited_without_returns() -> None:
    trials = [
        {"score_quantile": 0.95, "passes_support": True},
        {"score_quantile": 0.975, "passes_support": True},
        {"score_quantile": 0.99, "passes_support": False},
    ]
    assert clv.clvr._selected_support_quantile(trials) == 0.975


def test_frozen_clv_support_artifact_keeps_outcomes_sealed() -> None:
    result = json.loads(
        Path(
            "results/cross_collateral_liquidity_vacuum_support_2026-07-14.json"
        ).read_text()
    )
    assert result["protocol"]["outcomes_opened_for_clv"] is False
    assert result["protocol"]["support_rejected"] is False
    assert result["all_support_gates_pass"] is True
    assert result["support_calibration"]["selected_score_quantile"] == 0.975
    assert result["support"]["nonoverlap_total"] == 521
    assert result["support"]["by_quarter"] == {
        "q1": 91,
        "q2": 129,
        "q3": 148,
        "q4": 153,
    }
    assert result["support"]["h1"] == 220
    assert result["support"]["h2"] == 301
    assert result["scheduled_side_counts"] == {"long": 270, "short": 251}
    for key in (
        "preregistration_source",
        "preregistration_document",
        "feature_source",
        "scheduler_source",
    ):
        path = Path(result["frozen_artifacts"][key])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == result["frozen_artifacts"][f"{key}_sha256"]
