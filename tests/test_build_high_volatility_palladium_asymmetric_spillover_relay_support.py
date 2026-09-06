import json

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_palladium_asymmetric_spillover_relay_support as b


def _equity(dates, adjusted):
    return pd.DataFrame({"session_date": pd.to_datetime(dates), "adjusted_close": adjusted})


def test_yahoo_adjusted_close_and_official_common_sessions():
    timestamps = [int(pd.Timestamp(d, tz="America/New_York").timestamp()) for d in ("2024-01-02", "2024-01-03")]
    result = {
        "meta": {"symbol": "PALL", "exchangeTimezoneName": "America/New_York"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{"volume": [10, 20], "close": [101, 102], "open": [100, 101], "low": [99, 100], "high": [102, 103]}],
            "adjclose": [{"adjclose": [91.0, 92.0]}],
        },
    }
    stable, frame, _ = b.normalize_yahoo_chart(json.dumps({"chart": {"result": [result], "error": None}}).encode(), "PALL")
    assert frame.adjusted_close.tolist() == [91.0, 92.0]
    assert json.loads(stable)["adjclose"]["adjclose"] == [91.0, 92.0]
    dates = ["2023-07-03", "2023-07-05"]
    panel = b.build_equity_panel({symbol: _equity(dates, [100.0, 101.0]) for symbol in b.SYMBOLS})
    assert panel.cash_close_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == ["2023-07-03T17:00Z", "2023-07-05T20:00Z"]


def test_btc_grid_and_strict_prior_beta():
    start = pd.Timestamp("2024-01-01T00:00Z"); end = start + pd.Timedelta(minutes=3)
    raw = pd.DataFrame({"ts": pd.date_range(start, end, freq="1min", inclusive="left"), "open": [1, 2, 3], "close": [2, 3, 4]})
    assert len(b.normalize_btc_bars(raw, start, end)) == 3
    with pytest.raises(RuntimeError, match="exact requested 1m grid"):
        b.normalize_btc_bars(raw.iloc[[0, 2]], start, end)
    benchmark = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0]); equity = pd.Series([3.0, 1.0, 5.0, 2.0, 1000.0])
    beta = b.strict_prior_beta(benchmark, equity, lookback=4, minimum=3)
    expected = np.cov(benchmark.iloc[:4], equity.iloc[:4], ddof=0)[0, 1] / np.var(benchmark.iloc[:4])
    assert beta.iloc[4] == pytest.approx(expected)


def _signal_frame():
    return pd.DataFrame({
        "pall_residual": [0.4, -0.5, -0.6, 0.2],
        "prior_pall_residual": [np.nan, 0.4, -0.5, -0.6],
        "pall_return": [0.3, -0.2, 0.1, -0.4],
        "btc_variation_rank": [0.9, 0.9, 0.9, 0.9],
    })


def test_transition_signal_and_controls_are_frozen():
    frame = _signal_frame()
    active, side = b._signal(frame, "primary")
    assert active.tolist() == [False, True, False, True]
    assert side.tolist() == [1, -1, -1, 1]
    stale, _ = b._signal(frame, "one_session_stale_transition")
    assert stale.tolist() == [False, False, True, False]
    raw, _ = b._signal(frame, "raw_pall_return_transition")
    assert raw.tolist() == [False, True, True, True]
    level, _ = b._signal(frame, "residual_level_without_transition")
    assert level.tolist() == [True, True, True, True]
    _, flipped = b._signal(frame, "direction_flip")
    assert flipped.tolist() == [-1, 1, 1, -1]
    assert len(b.CONTROLS) == 6


def test_clock_uses_cash_close_latency_half_open_and_splits():
    frame = pd.DataFrame({
        "session_date": pd.to_datetime(["2023-07-03", "2023-07-04", "2023-07-05"]),
        "cash_close_time": pd.to_datetime(["2023-07-03T17:00Z", "2023-07-04T17:00Z", "2023-07-05T17:00Z"]),
        "pall_residual": [0.4, -0.5, 0.6], "prior_pall_residual": [-0.2, 0.4, -0.5],
        "pall_return": [0.3, -0.2, 0.1], "pall_beta": [1.1] * 3,
        "btc_variation_rank": [0.9] * 3, "btc_realized_variation": [0.01] * 3,
    })
    clock = b.build_clock(frame)
    assert len(clock) == 3
    assert clock.entry_time.dt.strftime("%Y-%m-%dT%H:%MZ").iloc[0] == "2023-07-03T17:10Z"
    assert clock.side.tolist() == [1, -1, 1]


def test_support_gates_fail_closed():
    assert b.support_stats(pd.DataFrame(columns=b.CLOCK_COLUMNS), "train")["events"] == 0
    assert b.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
