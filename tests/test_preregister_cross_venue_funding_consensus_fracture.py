from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training import preregister_cross_venue_funding_consensus_fracture as cfcf


def _small_cfg() -> cfcf.Config:
    return replace(
        cfcf.Config(),
        crowding_quantile=0.50,
        crowding_baseline_events=4,
        crowding_min_periods=2,
        hold_bars=2,
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_side_share=0.0,
        minimum_branch_share=0.0,
    )


def _settlements(*, rich: bool = True) -> pd.DataFrame:
    score = 10.0 if rich else -10.0
    sign = 1.0 if rich else -1.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=8, freq="8h"),
            "signal_date": pd.date_range(
                "2023-01-01 00:55", periods=8, freq="8h"
            ),
            "premium_z": [1.0, 1.0, 1.0, 1.0, sign, 0.0, 0.0, 0.0],
            "funding_z": [1.0, 1.0, 1.0, 1.0, sign, 0.0, 0.0, 0.0],
            "spread_agreement": [True, True, True, True, True, False, False, False],
            "crowding_score": [1.0, 1.0, 1.0, 1.0, score, 0.0, 0.0, 0.0],
        }
    )


def test_lagged_robust_score_is_prefix_invariant() -> None:
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    baseline = cfcf.lagged_robust_zscore(values, window=4, minimum=2)
    changed = values.copy()
    changed.loc[5] = -1_000_000.0
    replay = cfcf.lagged_robust_zscore(changed, window=4, minimum=2)
    pd.testing.assert_series_equal(baseline.loc[:4], replay.loc[:4])


def test_rich_and_cheap_consensus_trade_convergence() -> None:
    rich = cfcf.classify_settlements(_settlements(), _small_cfg())
    assert rich.loc[4, "candidate"]
    assert rich.loc[4, "branch"] == "bybit_rich"
    assert rich.loc[4, "side"] == -1

    cheap = cfcf.classify_settlements(_settlements(rich=False), _small_cfg())
    assert cheap.loc[4, "candidate"]
    assert cheap.loc[4, "branch"] == "bybit_cheap"
    assert cheap.loc[4, "side"] == 1


def test_projection_places_signal_at_settlement_hour_close() -> None:
    state = cfcf.classify_settlements(_settlements().iloc[:5], _small_cfg())
    market = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=400, freq="5min"),
            "quarantined": np.zeros(400, dtype=bool),
        }
    )
    signal = cfcf.project_to_market(state, market)
    signal_position = market.index[market["date"].eq("2023-01-02 08:55")][0]
    origin_position = market.index[market["date"].eq("2023-01-02 08:00")][0]
    assert signal.loc[signal_position, "side"] == -1
    assert signal.loc[signal_position, "origin_position"] == origin_position


def test_frozen_hold_enters_after_hour_close_and_exits_at_next_settlement() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=97, freq="5min"),
            "quarantined": np.zeros(97, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(97, dtype=np.int8),
            "hold_bars": np.zeros(97, dtype=np.int16),
            "branch": pd.Series("none", index=market.index, dtype="string"),
            "origin_position": np.full(97, -1, dtype=np.int64),
        }
    )
    signal_position = market.index[market["date"].eq("2023-01-01 00:55")][0]
    signal.loc[signal_position, ["side", "hold_bars", "branch", "origin_position"]] = [
        -1,
        84,
        "bybit_rich",
        0,
    ]
    schedule = cfcf.nonoverlapping_cfcf_schedule(
        signal,
        market,
        start="2023-01-01",
        end="2023-01-02",
    )
    assert schedule.loc[0, "signal_date"] == "2023-01-01 00:55:00"
    assert schedule.loc[0, "entry_date"] == "2023-01-01 01:00:00"
    assert schedule.loc[0, "exit_date"] == "2023-01-01 08:00:00"


def test_schedule_requires_settlement_origin_inside_split() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-06-30 23:45", periods=12, freq="5min"
            ),
            "quarantined": np.zeros(12, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": [0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
            "hold_bars": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            "branch": [
                "none",
                "none",
                "none",
                "bybit_cheap",
                "none",
                "bybit_rich",
                "none",
                "none",
                "none",
                "none",
                "none",
                "none",
            ],
            "origin_position": [-1, -1, -1, 1, -1, 3, -1, -1, -1, -1, -1, -1],
        }
    )
    schedule = cfcf.nonoverlapping_cfcf_schedule(
        signal,
        market,
        start="2023-07-01",
        end="2024-01-01",
    )
    assert schedule["signal_position"].tolist() == [5]
    assert schedule["origin_position"].tolist() == [3]


def test_support_rule_selects_strictest_passing_quantile() -> None:
    trials = [
        {"crowding_quantile": 0.80, "passes_support": True},
        {"crowding_quantile": 0.90, "passes_support": True},
        {"crowding_quantile": 0.925, "passes_support": False},
    ]
    assert cfcf._selected_support_quantile(trials) == 0.90


def test_frozen_support_artifact_keeps_outcomes_sealed() -> None:
    result = json.loads(
        Path(
            "results/cross_venue_funding_consensus_fracture_support_2026-07-14.json"
        ).read_text()
    )
    assert result["protocol"]["outcomes_opened_for_cfcf"] is False
    assert result["all_support_gates_pass"] is True
    assert result["support_calibration"]["selected_crowding_quantile"] == 0.90
    assert result["support"]["nonoverlap_total"] == 223
    assert result["support"]["by_year"] == {
        "2021": 46,
        "2022": 88,
        "2023": 89,
    }
    assert result["support"]["2023_h1"] == 38
    assert result["support"]["2023_h2"] == 51
    assert result["scheduled_branch_counts"] == {
        "bybit_cheap": 120,
        "bybit_rich": 103,
    }
    for key in (
        "preregistration_source",
        "preregistration_document",
        "scheduler_source",
    ):
        path = Path(result["frozen_artifacts"][key])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == result["frozen_artifacts"][f"{key}_sha256"]
