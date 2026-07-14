from __future__ import annotations

import pandas as pd
import pytest

from training import evaluate_um_forced_flow_reversion as ev


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_position": 0,
                "entry_position": 1,
                "exit_position": 37,
                "signal_date": "2020-01-01 00:00:00",
                "entry_date": "2020-01-01 00:05:00",
                "exit_date": "2020-01-01 03:05:00",
                "side": 1,
                "branch": "umfr36",
                "hold_bars": 36,
            },
            {
                "signal_position": 37,
                "entry_position": 38,
                "exit_position": 74,
                "signal_date": "2020-01-01 03:05:00",
                "entry_date": "2020-01-01 03:10:00",
                "exit_date": "2020-01-01 06:10:00",
                "side": -1,
                "branch": "umfr36",
                "hold_bars": 36,
            },
        ],
        columns=ev.CLOCK_COLUMNS,
    )


def test_slice_schedule_requires_signal_entry_and_exit_inside_split() -> None:
    schedule = _schedule()
    sliced = ev.slice_schedule(schedule, start="2020-01-01", end="2020-01-01 03:10:00")
    assert len(sliced) == 1
    assert sliced.iloc[0]["entry_position"] == 1
    with pytest.raises(ValueError, match="start must precede"):
        ev.slice_schedule(schedule, start="2020-01-02", end="2020-01-01")


def test_simulation_uses_36_bar_hold_and_excludes_exit_bar_extreme() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        }
    )
    frame.loc[37, "open"] = 110.0
    frame.loc[37, "low"] = (
        1.0  # must not affect held-path MDD after scheduled exit open
    )
    funding = pd.DataFrame(
        {"funding_time_ms": [], "funding_time": [], "funding_rate": []}
    ).astype({"funding_time_ms": "int64", "funding_rate": "float64"})
    schedule = _schedule().iloc[[0]].copy()
    cfg = ev.EvaluationConfig(cluster_permutations=10)
    metrics = ev.simulate_funding_schedule(
        frame,
        funding,
        schedule,
        start="2020-01-01",
        end="2020-01-02",
        cfg=cfg,
    )
    assert metrics["trade_count"] == 1
    assert metrics["absolute_return_pct"] > 4.0
    assert metrics["strict_mdd_pct"] < 2.0

    broken = schedule.copy()
    broken.loc[0, "exit_position"] = 35
    with pytest.raises(ValueError, match="36-bar hold"):
        ev.simulate_funding_schedule(
            frame, funding, broken, start="2020-01-01", end="2020-01-02", cfg=cfg
        )


def test_validate_evaluation_config_is_frozen() -> None:
    ev._validate_evaluation_config(ev.EvaluationConfig())
    with pytest.raises(ValueError, match="frozen"):
        ev._validate_evaluation_config(ev.EvaluationConfig(leverage=1.0))
