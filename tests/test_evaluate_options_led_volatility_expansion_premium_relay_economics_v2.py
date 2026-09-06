from __future__ import annotations

import math

import pandas as pd

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v2 as e,
)


def market(dates: pd.DatetimeIndex, opens: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": opens,
            "low": opens,
            "close": opens,
        }
    )


def no_funding() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "funding_rate", "mark_price"])


def test_back_to_back_trade_exits_before_same_open_reentry() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    dates = pd.date_range(start, start + pd.Timedelta(minutes=10), freq="5min")
    end = start + pd.Timedelta(days=365)
    candles = market(dates, [100.0, 110.0, 100.0])
    clock = pd.DataFrame(
        {
            "entry_time": [dates[0], dates[1]],
            "exit_time": [dates[1], dates[2]],
            "side": [1, -1],
        }
    )

    result = e.simulate(clock, candles, no_funding(), start, end, cost=0.0)

    expected = 1.05 * (1.0 + 0.5 * (1.0 - 100.0 / 110.0))
    assert result["trades"] == 2
    assert math.isclose(result["final_equity"], expected, rel_tol=0, abs_tol=1e-12)


def test_fixed_quantity_funding_cash_uses_settlement_mark_and_half_open_interval() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=10)
    dates = pd.date_range(start, end, freq="5min", inclusive="both")
    candles = market(dates, [100.0, 100.0, 100.0])
    funding = pd.DataFrame(
        {
            "date": [dates[0], dates[2]],
            "funding_rate": [0.01, 0.50],
            "mark_price": [100.0, 100.0],
        }
    )
    clock = pd.DataFrame(
        {"entry_time": [dates[0]], "exit_time": [dates[2]], "side": [1]}
    )

    result = e.simulate(clock, candles, funding, start, end, cost=0.0)

    assert math.isclose(result["final_equity"], 0.995, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        result["trade_rows"][0]["funding_cash_over_pre_equity"],
        -0.005,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_strict_mdd_marks_favorable_then_adverse_held_bar_path() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=5)
    dates = pd.date_range(start, end, freq="5min", inclusive="both")
    candles = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0],
            "high": [120.0, 100.0],
            "low": [80.0, 100.0],
            "close": [100.0, 100.0],
        }
    )
    clock = pd.DataFrame(
        {"entry_time": [dates[0]], "exit_time": [dates[1]], "side": [1]}
    )

    result = e.simulate(clock, candles, no_funding(), start, end, cost=0.0)

    assert math.isclose(
        result["strict_mdd_pct"], 100.0 * (1.0 - 0.9 / 1.1), abs_tol=1e-12
    )


def test_v1_failure_and_novelty_authorize_train_without_loading_prices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        e,
        "load_sources",
        lambda *_: (_ for _ in ()).throw(AssertionError("prices must stay sealed")),
    )

    novelty = e.verify("train")

    assert novelty["advance_to_economic_outcomes"] is True
