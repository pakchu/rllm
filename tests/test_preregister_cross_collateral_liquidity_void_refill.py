from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training import preregister_cross_collateral_liquidity_void_refill as clvr


def _small_cfg() -> clvr.Config:
    return replace(
        clvr.Config(),
        response_bars=2,
        robust_baseline_bars=4,
        robust_min_periods=2,
        score_baseline_bars=4,
        score_min_periods=2,
        score_quantile=0.50,
        hold_bars=2,
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_half=0,
        minimum_nonoverlap_per_quarter=0,
        minimum_side_share=0.0,
        minimum_branch_share=0.0,
    )


def _geometry_frame(*, refill: bool) -> pd.DataFrame:
    rows = 7
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "close": [100.0] * 6 + [101.0],
            "quote_asset_volume": [1_000.0] * rows,
            "taker_buy_quote": [750.0] * rows,
            "source_complete": [True] * rows,
        }
    )
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            for distance in range(1, 6):
                frame[f"{venue}_depth_{side}{distance}"] = 100.0 * distance
    frame.loc[6, "cm_depth_p1"] = 200.0 if refill else 50.0
    return frame


def test_cross_collateral_response_orients_void_and_refill() -> None:
    cfg = replace(_small_cfg(), response_bars=6)
    void = clvr.build_features(_geometry_frame(refill=False), cfg)
    assert void.loc[6, "direction"] == 1
    assert void.loc[6, "level_response"] > 0.0
    assert void.loc[6, "shape_response"] > 0.0

    refill = clvr.build_features(_geometry_frame(refill=True), cfg)
    assert refill.loc[6, "direction"] == 1
    assert refill.loc[6, "level_response"] < 0.0
    assert refill.loc[6, "shape_response"] < 0.0


def _manual_features(*, refill: bool) -> pd.DataFrame:
    sign = -1.0 if refill else 1.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=8, freq="5min"),
            "clean": [True] * 8,
            "flow_aligned": [True] * 8,
            "direction": [1] * 8,
            "response_return_z": [1.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0],
            "level_response_z": [1.0, 1.0, 1.0, 1.0, 10.0 * sign, 0.0, 0.0, 0.0],
            "shape_response_z": [1.0, 1.0, 1.0, 1.0, 10.0 * sign, 0.0, 0.0, 0.0],
        }
    )


def test_void_follows_shock_and_refill_fades_shock() -> None:
    void = clvr.classify_features(_manual_features(refill=False), _small_cfg())
    assert void.loc[4, "candidate"]
    assert void.loc[4, "branch"] == "void"
    assert void.loc[4, "side"] == 1

    refill = clvr.classify_features(_manual_features(refill=True), _small_cfg())
    assert refill.loc[4, "candidate"]
    assert refill.loc[4, "branch"] == "refill"
    assert refill.loc[4, "side"] == -1


def test_lagged_robust_score_is_prefix_invariant() -> None:
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    baseline = clvr.lagged_robust_zscore(values, window=4, minimum=2)
    changed = values.copy()
    changed.loc[5] = -1_000_000.0
    replay = clvr.lagged_robust_zscore(changed, window=4, minimum=2)
    pd.testing.assert_series_equal(baseline.loc[:4], replay.loc[:4])


def test_support_rule_selects_strictest_passing_quantile() -> None:
    trials = [
        {"score_quantile": 0.95, "passes_support": True},
        {"score_quantile": 0.975, "passes_support": True},
        {"score_quantile": 0.99, "passes_support": False},
    ]
    assert clvr._selected_support_quantile(trials) == 0.975


def test_support_rule_returns_none_when_every_quantile_fails() -> None:
    trials = [
        {"score_quantile": 0.95, "passes_support": False},
        {"score_quantile": 0.975, "passes_support": False},
    ]
    assert clvr._selected_support_quantile(trials) is None


def test_frozen_support_artifact_rejects_without_opening_outcomes() -> None:
    result = json.loads(
        Path(
            "results/cross_collateral_liquidity_void_refill_support_2026-07-14.json"
        ).read_text()
    )
    assert result["protocol"]["outcomes_opened_for_clvr"] is False
    assert result["protocol"]["support_rejected"] is True
    assert result["all_support_gates_pass"] is False
    assert result["support_calibration"]["selected_score_quantile"] is None
    assert result["support"] is None
    assert [
        (trial["score_quantile"], trial["nonoverlap_total"])
        for trial in result["support_calibration"]["trials"]
    ] == [
        (0.9, 1_936),
        (0.925, 1_535),
        (0.95, 1_089),
        (0.975, 618),
        (0.99, 282),
        (0.995, 156),
    ]
    for key in (
        "preregistration_source",
        "preregistration_document",
        "scheduler_source",
    ):
        path = Path(result["frozen_artifacts"][key])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == result["frozen_artifacts"][f"{key}_sha256"]
