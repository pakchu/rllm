from __future__ import annotations

import numpy as np
import pandas as pd

from training.backtest_added_alpha_month import (
    _fixed_hold_arrays,
    _interval_slots,
    _strict_metric,
)


def _market(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-07-01", periods=rows, freq="5min")
    open_price = np.linspace(100.0, 104.0, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": open_price * 1.002,
            "low": open_price * 0.998,
            "close": open_price,
        }
    )


def test_interval_slots_use_absolute_timestamp_offset() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2026-07-01 20:50:00Z",
                "2026-07-01 20:55:00Z",
                "2026-07-01 21:00:00Z",
            ]
        )
    )
    assert _interval_slots(dates, 24, 11).tolist() == [False, True, False]


def test_fixed_hold_scheduler_suppresses_overlapping_trade() -> None:
    market = _market()
    signal = np.zeros(len(market), dtype=np.int8)
    signal[[10, 15, 24]] = -1
    result = _fixed_hold_arrays(
        market,
        signal,
        name="test",
        hold_bars=12,
        start=pd.Timestamp("2026-07-01"),
        end=pd.Timestamp("2026-07-02"),
    )
    assert len(result["trades"]) == 2
    assert result["skipped_overlap"] == 1
    assert [trade["signal_date"] for trade in result["trades"]] == [
        "2026-07-01 00:50:00",
        "2026-07-01 02:00:00",
    ]


def test_strict_metric_applies_upper_before_lower_same_bar() -> None:
    dates = pd.Series(pd.to_datetime(["2026-07-01 00:00:00"]))
    arrays = {
        "alpha": {
            "R": np.asarray([0.0]),
            "L": np.asarray([-0.10]),
            "H": np.asarray([0.20]),
            "trades": [],
        }
    }
    metric = _strict_metric(
        arrays,
        {"alpha": 1.0},
        dates=dates,
        start=pd.Timestamp("2026-07-01 00:00:00"),
        end=pd.Timestamp("2026-07-01 00:05:00"),
    )
    assert np.isclose(metric["strict_mdd_pct"], 25.0)
    assert metric["absolute_return_pct"] == 0.0
