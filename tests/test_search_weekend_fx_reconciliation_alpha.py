from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import search_weekend_fx_reconciliation_alpha as weekend_fx


def test_prior_event_zscore_is_prefix_invariant() -> None:
    values = pd.Series(np.sin(np.arange(200) / 9.0))
    first = weekend_fx.prior_event_zscore(values, lookback=52, min_observations=26)
    changed = values.copy()
    changed.iloc[120:] = 1e12
    second = weekend_fx.prior_event_zscore(changed, lookback=52, min_observations=26)
    np.testing.assert_allclose(first.iloc[:120], second.iloc[:120], equal_nan=True)


def test_prior_event_zscore_excludes_current_event() -> None:
    values = pd.Series(np.arange(30, dtype=float))
    score = weekend_fx.prior_event_zscore(values, lookback=10, min_observations=5)
    expected = (values.iloc[20] - values.iloc[10:20].mean()) / values.iloc[10:20].std(ddof=0)
    assert score.iloc[20] == pytest.approx(expected)


def test_safe_haven_gap_differential_uses_fixed_orientation() -> None:
    previous = pd.Series({f"close_{ticker}": 100.0 for ticker in weekend_fx.TICKERS})
    current = previous.copy()
    current["close_USDJPY"] = 99.0
    current["close_USDCHF"] = 99.0
    oriented = weekend_fx.oriented_fx_gap_returns(current, previous)
    differential = weekend_fx.safe_haven_gap_differential(oriented)
    assert oriented["USDJPY"] < 0.0
    assert oriented["USDCHF"] < 0.0
    assert differential < 0.0


def test_validate_market_source_rejects_malformed_ohlc() -> None:
    market = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01")],
            "tic": ["BTCUSDT"],
            "open": [100.0],
            "high": [99.0],
            "low": [98.0],
            "close": [100.0],
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        weekend_fx.validate_market_source(market)


def _synthetic_weekend_inputs(weeks: int = 28) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    market_rows: list[dict[str, object]] = []
    fx_rows: list[dict[str, object]] = []
    price = 100.0
    for week in range(weeks):
        monday = pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=week)
        friday = monday - pd.Timedelta(days=2, hours=2)  # Friday 22:00 -> Monday 00:00 = 50h.
        weekend_step = 0.001 + np.cos(week / 4.0) * 0.002
        for boundary, step in ((friday, weekend_step), (monday, -0.0005 + week * 1e-5)):
            market_rows.append({"date": boundary - pd.Timedelta("5min"), "close": price})
            market_rows.append({"date": boundary, "close": price})
            row: dict[str, object] = {
                "effective_time": boundary,
                "source_time": boundary - pd.Timedelta("1min"),
                "valid_hour": True,
            }
            shock = np.sin(week / 3.0) * 0.01 if boundary == monday else 0.0
            for index, ticker in enumerate(weekend_fx.TICKERS):
                group_sign = 1.0 if ticker in weekend_fx.SAFE_HAVENS else -1.0
                row[f"close_{ticker}"] = 100.0 + index + week * 0.001 + group_sign * shock
            fx_rows.append(row)
            price *= 1.0 + step
    market = pd.DataFrame(market_rows).sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    fx = pd.DataFrame(fx_rows).sort_values("effective_time").reset_index(drop=True)
    return market, dates, fx


def test_build_event_table_uses_friday_and_monday_completed_boundaries() -> None:
    market, dates, fx = _synthetic_weekend_inputs()
    events = weekend_fx.build_event_table(market, dates, fx)
    assert len(events) == 28
    assert events["closure_hours"].eq(50.0).all()
    assert (
        events["fx_source_time"]
        == events["effective_time"] - pd.Timedelta("1min")
    ).all()
    assert (
        events["btc_source_time"]
        == events["effective_time"] - pd.Timedelta("5min")
    ).all()
    assert int(events["eligible"].sum()) == 2
    assert events.loc[events["eligible"], "effective_time"].dt.dayofweek.eq(0).all()


def test_build_event_table_excludes_nonweekend_outage() -> None:
    market, dates, fx = _synthetic_weekend_inputs(weeks=28)
    outage_time = pd.Timestamp("2020-08-05 12:00")
    prior_time = outage_time - pd.Timedelta(hours=50)
    extra_market = pd.DataFrame(
        {
            "date": [prior_time - pd.Timedelta("5min"), prior_time, outage_time - pd.Timedelta("5min"), outage_time],
            "close": [100.0, 100.0, 101.0, 101.0],
        }
    )
    market = pd.concat([market, extra_market], ignore_index=True).sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    extra_fx = []
    for boundary in (prior_time, outage_time):
        row = {
            "effective_time": boundary,
            "source_time": boundary - pd.Timedelta("1min"),
            "valid_hour": True,
        }
        row.update({f"close_{ticker}": 100.0 for ticker in weekend_fx.TICKERS})
        extra_fx.append(row)
    fx = pd.concat([fx, pd.DataFrame(extra_fx)], ignore_index=True).sort_values("effective_time").reset_index(drop=True)
    events = weekend_fx.build_event_table(market, dates, fx)
    assert outage_time not in set(events["effective_time"])


def test_policy_trades_toward_reconciliation_residual() -> None:
    state = pd.DataFrame(
        {
            "eligible": [True, True, True, False],
            "reconciliation_residual": [2.0, -3.0, 0.0, 5.0],
        }
    )
    long_active, short_active = weekend_fx.policy_masks(state)
    assert np.flatnonzero(long_active).tolist() == [0]
    assert np.flatnonzero(short_active).tolist() == [1]


def test_lag_sparse_event_masks_uses_previous_closure_side() -> None:
    event = np.array([False, True, False, False, True, False, True])
    long_active = np.array([False, True, False, False, False, False, True])
    short_active = np.array([False, False, False, False, True, False, False])
    lag_long, lag_short = weekend_fx.lag_sparse_event_masks(long_active, short_active, event)
    assert np.flatnonzero(lag_long).tolist() == [4]
    assert np.flatnonzero(lag_short).tolist() == [6]


def test_support_counts_are_nonoverlapping_and_split_contained(monkeypatch) -> None:
    monkeypatch.setitem(
        weekend_fx.WINDOWS,
        "sample",
        ("2023-01-01", "2023-01-01 01:00"),
    )
    dates = pd.Series(pd.date_range("2023-01-01", periods=13, freq="5min"))
    long_active = np.array([True, True, False, False, False, False, True, False, False, False, False, False, False])
    short_active = np.array([False, False, False, False, True, False, False, False, False, False, True, False, False])
    counts = weekend_fx.support_counts(
        dates,
        long_active,
        short_active,
        window="sample",
        hold_bars=2,
    )
    assert counts == {
        "raw": 5,
        "raw_long": 3,
        "raw_short": 2,
        "strict_executable": 2,
        "strict_executable_long": 1,
        "strict_executable_short": 1,
    }


def test_support_only_cannot_open_outcomes_or_write(monkeypatch, tmp_path: Path) -> None:
    market = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01")],
            "tic": ["BTCUSDT"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    dates = pd.Series([pd.Timestamp("2023-01-01")])
    fx = pd.DataFrame({"valid_hour": [True]})
    events = pd.DataFrame(
        {
            "eligible": [True],
            "effective_time": [pd.Timestamp("2023-01-01")],
            "fx_source_time": [pd.Timestamp("2022-12-31 23:59")],
        }
    )
    state = pd.DataFrame(
        {
            "eligible": [True],
            "reconciliation_residual": [1.0],
        }
    )
    monkeypatch.setattr(weekend_fx, "load_market_before", lambda *args: (market, dates))
    monkeypatch.setattr(weekend_fx, "read_completed_fx_hours_before", lambda *args: fx)
    monkeypatch.setattr(weekend_fx, "build_event_table", lambda *args: events)
    monkeypatch.setattr(weekend_fx, "build_state", lambda *args: state)
    monkeypatch.setattr(weekend_fx, "policy_masks", lambda *args: (np.array([True]), np.array([False])))
    monkeypatch.setattr(
        weekend_fx,
        "support_counts",
        lambda *args, window, **kwargs: {
            "raw": 100,
            "raw_long": 50,
            "raw_short": 50,
            "strict_executable": 100 if window == "fit" else 30,
            "strict_executable_long": 50 if window == "fit" else 15,
            "strict_executable_short": 50 if window == "fit" else 15,
        },
    )
    monkeypatch.setattr(weekend_fx, "RESULT_PATH", tmp_path / "forbidden.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed the outcome boundary")

    monkeypatch.setattr(weekend_fx, "_future_extreme", forbidden)
    monkeypatch.setattr(weekend_fx, "simulate", forbidden)
    output = weekend_fx.run(support_only=True)
    assert output["outcomes_opened"] is False
    assert output["support_passed"] is True
    assert not weekend_fx.RESULT_PATH.exists()
