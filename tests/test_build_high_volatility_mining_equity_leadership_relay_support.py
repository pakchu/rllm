import json

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_mining_equity_leadership_relay_support as b


def test_normalize_yahoo_chart_reads_adjusted_close_and_canonicalizes_payload():
    timestamps = [int(pd.Timestamp(date, tz="America/New_York").timestamp()) for date in ("2024-01-02", "2024-01-03")]
    result = {
        "meta": {"symbol": "RIOT", "exchangeTimezoneName": "America/New_York"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{"volume": [10, 20], "close": [101, 102], "open": [100, 101], "low": [99, 100], "high": [102, 103]}],
            "adjclose": [{"adjclose": [91.0, 92.0]}],
        },
    }
    payload = json.dumps({"chart": {"result": [result], "error": None}}).encode()
    stable, frame, metadata = b.normalize_yahoo_chart(payload, "RIOT")
    assert frame.adjusted_close.tolist() == [91.0, 92.0]
    assert metadata["adjusted_close_read"] is True
    assert json.loads(stable)["adjclose"]["adjclose"] == [91.0, 92.0]


def _equity(dates, adjusted):
    return pd.DataFrame({"session_date": pd.to_datetime(dates), "adjusted_close": adjusted})


def test_common_panel_uses_official_early_close_and_rejects_missing_session():
    dates = ["2023-07-03", "2023-07-05"]
    equities = {symbol: _equity(dates, [100.0, 101.0]) for symbol in b.SYMBOLS}
    panel = b.build_equity_panel(equities)
    assert panel.cash_close_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == [
        "2023-07-03T17:00Z",
        "2023-07-05T20:00Z",
    ]
    broken = dict(equities)
    broken["MARA"] = _equity(["2023-07-03"], [100.0])
    with pytest.raises(RuntimeError, match="share every official common session"):
        b.build_equity_panel(broken)


def test_btc_normalization_requires_exact_read_only_grid():
    start = pd.Timestamp("2024-01-01T00:00Z")
    end = start + pd.Timedelta(minutes=3)
    raw = pd.DataFrame({"ts": pd.date_range(start, end, freq="1min", inclusive="left"), "open": [1, 2, 3], "close": [2, 3, 4]})
    assert b.normalize_btc_bars(raw, start, end).columns.tolist() == ["ts", "open", "close"]
    with pytest.raises(RuntimeError, match="exact requested 1m grid"):
        b.normalize_btc_bars(raw.iloc[[0, 2]], start, end)


def test_strict_prior_beta_excludes_current_pair_and_uses_frozen_window():
    btc = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
    miner = pd.Series([3.0, 1.0, 5.0, 2.0, 1000.0])
    actual = b.strict_prior_beta(btc, miner, lookback=4, minimum=3)
    assert actual.iloc[:3].isna().all()
    expected3 = np.cov(btc.iloc[:3], miner.iloc[:3], ddof=0)[0, 1] / np.var(btc.iloc[:3])
    expected4 = np.cov(btc.iloc[:4], miner.iloc[:4], ddof=0)[0, 1] / np.var(btc.iloc[:4])
    assert actual.iloc[3] == pytest.approx(expected3)
    assert actual.iloc[4] == pytest.approx(expected4)


def test_residual_ranks_are_strict_prior_and_use_frozen_windows(monkeypatch):
    calls = []

    def fake_rank(values, lookback, minimum):
        calls.append((values.name, lookback, minimum))
        return pd.Series(0.9, index=values.index)

    monkeypatch.setattr(b, "strict_prior_midrank", fake_rank)
    monkeypatch.setattr(
        b,
        "strict_prior_beta",
        lambda btc, miner, lookback, minimum: pd.Series(1.0, index=btc.index),
    )
    sessions = pd.DataFrame(
        {
            "session_date": pd.date_range("2024-01-02", periods=21, freq="B"),
            "cash_close_time": pd.date_range("2024-01-02T21:00Z", periods=21, freq="24h"),
            "riot_adjusted_close": np.exp(np.arange(21) * 0.01),
            "mara_adjusted_close": np.exp(np.arange(21) * 0.02),
            "hut_adjusted_close": np.exp(np.arange(21) * -0.01),
        }
    )
    start = sessions.cash_close_time.min() - pd.Timedelta(hours=24)
    end = sessions.cash_close_time.max()
    index = pd.date_range(start, end, freq="1min", inclusive="left")
    bars = pd.DataFrame({"ts": index, "open": 100.0, "close": 100.0 + np.arange(len(index)) / 10000})
    features = b.build_features(sessions, bars)
    assert features.leadership_residual.iloc[0] != features.leadership_residual.iloc[0]
    assert calls == [
        ("leadership_residual", 270, 180),
        ("btc_realized_variation", 270, 180),
        ("raw_equal_mean_miner_return", 270, 180),
        ("mara_residual", 270, 180),
    ]


def _signal_frame():
    return pd.DataFrame(
        {
            "leadership_residual": [0.4, -0.5, 0.6],
            "magnitude_rank": [0.9, 0.9, 0.9],
            "btc_variation_rank": [0.7, 0.6, 0.8],
            "raw_equal_mean_miner_return": [-0.4, 0.5, -0.6],
            "raw_magnitude_rank": [0.9, 0.9, 0.9],
            "mara_residual": [0.3, -0.2, 0.1],
            "mara_magnitude_rank": [0.9, 0.9, 0.9],
        }
    )


def test_primary_and_all_six_diagnostics_have_frozen_direction_and_gates():
    frame = _signal_frame()
    active, side = b._signal(frame, "primary")
    assert active.tolist() == [True, False, True]
    assert side.tolist() == [1, -1, 1]
    active, _ = b._signal(frame, "no_btc_volatility_gate")
    assert active.tolist() == [True, True, True]
    _, flipped = b._signal(frame, "direction_flip")
    assert flipped.tolist() == [-1, 1, -1]
    _, raw = b._signal(frame, "raw_equal_mean_miner_return")
    assert raw.tolist() == [-1, 1, -1]
    _, mara = b._signal(frame, "mara_only_residual")
    assert mara.tolist() == [1, -1, 1]
    _, forced = b._signal(frame, "same_clock_forced_long")
    assert forced.tolist() == [1, 1, 1]
    stale_active, stale = b._signal(frame, "one_session_stale_residual")
    assert stale_active.tolist() == [False, False, True]
    assert stale.iloc[2] == -1
    assert len(b.CONTROLS) == 6


def test_clock_uses_positive_side_early_close_global_half_open_and_fixed_splits():
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2023-07-03", "2023-07-04", "2023-07-05"]),
            "cash_close_time": pd.to_datetime(["2023-07-03T17:00Z", "2023-07-04T17:00Z", "2023-07-05T17:00Z"]),
            "leadership_residual": [0.4, -0.5, -0.6],
            "magnitude_rank": [0.9] * 3,
            "btc_variation_rank": [0.9] * 3,
            "riot_beta": [1.1] * 3,
            "mara_beta": [1.2] * 3,
            "hut_beta": [1.3] * 3,
            "riot_residual": [0.1] * 3,
            "mara_residual": [0.2] * 3,
            "hut_residual": [0.3] * 3,
            "btc_realized_variation": [0.01] * 3,
            "raw_equal_mean_miner_return": [0.1] * 3,
            "raw_magnitude_rank": [0.9] * 3,
            "mara_magnitude_rank": [0.9] * 3,
        }
    )
    clock = b.build_clock(frame)
    assert len(clock) == 3  # equality at the prior exit is admitted by [entry, exit)
    assert clock.entry_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist()[0] == "2023-07-03T17:10Z"
    assert clock.side.tolist() == [1, -1, -1]
    assert clock.split.unique().tolist() == ["train"]


def test_support_stats_and_gates_are_fail_closed_for_empty_split():
    empty = pd.DataFrame(columns=b.CLOCK_COLUMNS)
    assert b.support_stats(empty, "train") == {
        "events": 0,
        "longs": 0,
        "shorts": 0,
        "minority_side_share": 0.0,
        "max_month_share": 0.0,
    }
    assert b.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
