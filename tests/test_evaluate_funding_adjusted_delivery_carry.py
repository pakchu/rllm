from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from training import evaluate_funding_adjusted_delivery_carry as evaluate


def test_quarterly_parser_physically_stops_before_cutoff_values(tmp_path) -> None:
    path = tmp_path / "quarter.csv.gz"
    header = (
        "open_time,um_open,um_high,um_low,um_close,um_ohlc_valid,"
        "source_complete,delivery_time,contract_segment\n"
    )
    with gzip.open(path, "wt") as handle:
        handle.write(header)
        handle.write(
            "2022-12-31 23:55:00+00:00,100,101,99,100,True,True,"
            "2023-03-31 08:00:00+00:00,20230331\n"
        )
        handle.write(
            "2023-01-01 00:00:00+00:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,"
            "NOT_PARSED,True,True,2023-03-31 08:00:00+00:00,20230331\n"
        )
    frame = evaluate._parse_quarterly_before(path, pd.Timestamp("2023-01-01", tz="UTC"))
    assert len(frame) == 1
    assert frame.iloc[0]["quarter_open"] == 100.0


def test_perpetual_parser_physically_stops_before_cutoff_values(tmp_path) -> None:
    path = tmp_path / "perp.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        for minute in range(55, 60):
            handle.write(f"2022-12-31 23:{minute}:00,100,101,99,100\n")
        handle.write("2023-01-01 00:00:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
    frame = evaluate._parse_perpetual_before(path, pd.Timestamp("2023-01-01", tz="UTC"))
    assert len(frame) == 1
    assert frame.iloc[0]["minute_rows"] == 5


def test_funding_parser_physically_stops_before_cutoff_values(tmp_path) -> None:
    path = tmp_path / "funding.csv.gz"
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    with gzip.open(path, "wt") as handle:
        handle.write(
            "funding_time_ms,funding_time_utc,funding_rate,settlement_mark_price\n"
        )
        handle.write("1672502400000,2022-12-31T16:00:00Z,0.0001,100\n")
        handle.write(f"{int(cutoff.timestamp()*1000)},NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
    frame = evaluate._parse_funding_before(path, cutoff)
    assert len(frame) == 1
    assert frame.iloc[0]["settlement_mark_price"] == 100.0


def _market() -> pd.DataFrame:
    times = pd.date_range("2022-01-01", periods=4, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "quarter_open": [100.0] * 4,
            "quarter_high": [110.0, 100.0, 100.0, 100.0],
            "quarter_low": [90.0, 100.0, 100.0, 100.0],
            "quarter_close": [100.0] * 4,
            "quarter_ohlc_valid": [True] * 4,
            "quarter_source_complete": [True] * 4,
            "delivery_time": [pd.Timestamp("2022-03-25 08:00Z")] * 4,
            "contract_segment": ["20220325"] * 4,
            "perp_open": [100.0] * 4,
            "perp_high": [110.0, 100.0, 100.0, 100.0],
            "perp_low": [90.0, 100.0, 100.0, 100.0],
            "perp_close": [100.0] * 4,
            "minute_rows": [5] * 4,
            "perp_source_complete": [True] * 4,
        }
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "control": ["primary"],
            "entry_time": [pd.Timestamp("2022-01-01 00:00Z")],
            "exit_time": [pd.Timestamp("2022-01-01 00:15Z")],
            "mandatory_exit_time": [pd.Timestamp("2022-03-24 08:00Z")],
            "delivery_time": [pd.Timestamp("2022-03-25 08:00Z")],
            "contract_segment": ["20220325"],
            "perpetual_side": [1],
            "quarterly_side": [-1],
        }
    )


def _funding(rate: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": [pd.Timestamp("2022-01-01 00:05Z")],
            "funding_rate": [rate],
            "settlement_mark_price": [100.0],
        }
    )


def test_strict_mdd_combines_independent_favorable_before_adverse_extrema() -> None:
    result = evaluate.simulate_schedule(
        _market(),
        _funding(),
        _schedule(),
        period_start=pd.Timestamp("2022-01-01 00:00Z"),
        period_end=pd.Timestamp("2022-01-02 00:00Z"),
        cost_rate=0.0,
    )
    # q=1/(100+100)=0.005. Independent favorable pair makes +10%, then
    # independent adverse pair makes -10%: drawdown is 20/110.
    assert result["strict_mdd_pct"] == pytest.approx(100.0 * 20.0 / 110.0)
    assert result["absolute_return_pct"] == pytest.approx(0.0)


def test_funding_cash_uses_fixed_quantity_and_position_side() -> None:
    long_perp = evaluate.simulate_schedule(
        _market(),
        _funding(0.001),
        _schedule(),
        period_start=pd.Timestamp("2022-01-01 00:00Z"),
        period_end=pd.Timestamp("2022-01-02 00:00Z"),
        cost_rate=0.0,
    )
    flipped = _schedule()
    flipped[["perpetual_side", "quarterly_side"]] *= -1
    short_perp = evaluate.simulate_schedule(
        _market(),
        _funding(0.001),
        flipped,
        period_start=pd.Timestamp("2022-01-01 00:00Z"),
        period_end=pd.Timestamp("2022-01-02 00:00Z"),
        cost_rate=0.0,
    )
    assert long_perp["funding_cash_pct_initial"] == pytest.approx(-0.05)
    assert short_perp["funding_cash_pct_initial"] == pytest.approx(0.05)


def test_exit_time_funding_is_excluded() -> None:
    funding = pd.DataFrame(
        {
            "funding_time": [pd.Timestamp("2022-01-01 00:15Z")],
            "funding_rate": [0.01],
            "settlement_mark_price": [100.0],
        }
    )
    result = evaluate.simulate_schedule(
        _market(),
        funding,
        _schedule(),
        period_start=pd.Timestamp("2022-01-01 00:00Z"),
        period_end=pd.Timestamp("2022-01-02 00:00Z"),
        cost_rate=0.0,
    )
    assert result["funding_cash_pct_initial"] == 0.0


def test_entry_time_funding_is_included() -> None:
    funding = pd.DataFrame(
        {
            "funding_time": [pd.Timestamp("2022-01-01 00:00Z")],
            "funding_rate": [0.001],
            "settlement_mark_price": [100.0],
        }
    )
    result = evaluate.simulate_schedule(
        _market(),
        funding,
        _schedule(),
        period_start=pd.Timestamp("2022-01-01 00:00Z"),
        period_end=pd.Timestamp("2022-01-02 00:00Z"),
        cost_rate=0.0,
    )
    assert result["funding_cash_pct_initial"] == pytest.approx(-0.05)


def test_two_leg_entry_and_exit_costs_use_actual_notional() -> None:
    result = evaluate.simulate_schedule(
        _market(),
        _funding(),
        _schedule(),
        period_start=pd.Timestamp("2022-01-01 00:00Z"),
        period_end=pd.Timestamp("2022-01-02 00:00Z"),
        cost_rate=0.001,
    )
    assert result["transaction_cost_pct_initial"] == pytest.approx(0.2)
    assert result["absolute_return_pct"] == pytest.approx(-0.2)


def test_weekly_cluster_signflip_is_deterministic() -> None:
    index = pd.date_range("2022-01-01", periods=40, freq="1D", tz="UTC")
    equity = pd.Series(1.0 + np.arange(40) * 0.001, index=index)
    first = evaluate.weekly_cluster_signflip(equity, permutations=1000, seed=7)
    second = evaluate.weekly_cluster_signflip(equity, permutations=1000, seed=7)
    assert first == second
    assert first["cluster_count"] > 0
