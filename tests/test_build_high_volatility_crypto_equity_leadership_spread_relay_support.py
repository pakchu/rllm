import json

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_crypto_equity_leadership_spread_relay_support as support


def test_normalize_yahoo_chart_reads_adjusted_close_and_canonicalizes_payload() -> None:
    timestamps = [
        int(pd.Timestamp(date, tz="America/New_York").timestamp())
        for date in ("2024-01-02", "2024-01-03")
    ]
    result = {
        "meta": {"symbol": "MSTR", "exchangeTimezoneName": "America/New_York"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "volume": [10, 20], "close": [101, 102], "open": [100, 101],
                "low": [99, 100], "high": [102, 103],
            }],
            "adjclose": [{"adjclose": [91.0, 92.0]}],
        },
    }
    payload = json.dumps({"chart": {"result": [result], "error": None}}).encode()
    stable, frame, metadata = support.normalize_yahoo_chart(payload, "MSTR")
    assert frame.adjusted_close.tolist() == [91.0, 92.0]
    assert metadata["adjusted_close_read"] is True
    assert json.loads(stable)["adjclose"]["adjclose"] == [91.0, 92.0]


def _equity(dates, adjusted):
    return pd.DataFrame({"session_date": pd.to_datetime(dates), "adjusted_close": adjusted})


def test_common_panel_uses_official_early_close_and_rejects_missing_session() -> None:
    dates = ["2023-07-03", "2023-07-05"]
    equities = {symbol: _equity(dates, [100.0, 101.0]) for symbol in support.SYMBOLS}
    panel = support.build_equity_panel(equities)
    assert panel.cash_close_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == [
        "2023-07-03T17:00Z", "2023-07-05T20:00Z",
    ]
    broken = dict(equities)
    broken["COIN"] = _equity(["2023-07-03"], [100.0])
    with pytest.raises(RuntimeError, match="share every official common session"):
        support.build_equity_panel(broken)


def test_btc_normalization_requires_exact_read_only_grid() -> None:
    start = pd.Timestamp("2024-01-01T00:00Z")
    end = start + pd.Timedelta(minutes=3)
    raw = pd.DataFrame({
        "ts": pd.date_range(start, end, freq="1min", inclusive="left"),
        "open": [1, 2, 3], "close": [2, 3, 4],
    })
    assert support.normalize_btc_bars(raw, start, end).columns.tolist() == ["ts", "open", "close"]
    with pytest.raises(RuntimeError, match="exact requested 1m grid"):
        support.normalize_btc_bars(raw.iloc[[0, 2]], start, end)


def test_strict_prior_scale_excludes_current_and_uses_frozen_window() -> None:
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 1000.0])
    actual = support.strict_prior_scale(values, lookback=4, minimum=3)
    assert actual.iloc[:3].isna().all()
    assert actual.iloc[3] == pytest.approx(np.std(values.iloc[:3], ddof=0))
    assert actual.iloc[4] == pytest.approx(np.std(values.iloc[:4], ddof=0))


def test_feature_ranks_are_strict_prior_and_use_frozen_windows(monkeypatch) -> None:
    calls = []

    def fake_rank(values, lookback, minimum):
        calls.append((values.name, lookback, minimum))
        return pd.Series(0.9, index=values.index)

    monkeypatch.setattr(support, "strict_prior_midrank", fake_rank)
    monkeypatch.setattr(
        support, "strict_prior_scale",
        lambda returns, lookback, minimum: pd.Series(0.1, index=returns.index),
    )
    sessions = pd.DataFrame({
        "session_date": pd.date_range("2024-01-02", periods=21, freq="B"),
        "cash_close_time": pd.date_range("2024-01-02T21:00Z", periods=21, freq="24h"),
        "mstr_adjusted_close": np.exp(np.arange(21) * 0.02),
        "coin_adjusted_close": np.exp(np.arange(21) * -0.01),
    })
    start = sessions.cash_close_time.min() - pd.Timedelta(hours=24)
    end = sessions.cash_close_time.max()
    index = pd.date_range(start, end, freq="1min", inclusive="left")
    bars = pd.DataFrame({"ts": index, "open": 100.0, "close": 100.01})
    features = support.build_features(sessions, bars)
    assert np.isnan(features.leadership_spread.iloc[0])
    assert calls == [
        ("leadership_spread", 270, 180),
        ("btc_realized_variation", 270, 180),
        ("raw_return_spread", 270, 180),
    ]


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_valid": [True] * 4,
        "leadership_spread": [0.1, 0.2, -0.3, -0.1],
        "magnitude_rank": [0.6, 0.8, 0.8, 0.7],
        "raw_return_spread": [-0.1, -0.2, 0.3, 0.1],
        "raw_magnitude_rank": [0.6, 0.8, 0.8, 0.7],
        "btc_variation_rank": [0.8] * 4,
    })


def test_primary_onset_and_all_controls_use_frozen_direction_and_gates() -> None:
    frame = _signal_frame()
    active, side = support._signal(frame, "primary")
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]
    _, raw = support._signal(frame, "raw_return_spread")
    assert raw.tolist() == [-1, -1, 1, 1]
    _, flipped = support._signal(frame, "direction_flip")
    assert flipped.tolist() == [-1, -1, 1, 1]
    _, forced = support._signal(frame, "same_clock_forced_long")
    assert forced.tolist() == [1, 1, 1, 1]
    assert len(support.CONTROLS) == 6


def test_clock_uses_early_close_and_fixed_split() -> None:
    frame = pd.DataFrame({
        "session_date": pd.to_datetime(["2023-07-03", "2023-07-05"]),
        "cash_close_time": pd.to_datetime(["2023-07-03T17:00Z", "2023-07-05T20:00Z"]),
        "source_valid": [True, True],
        "leadership_spread": [0.1, -0.3], "magnitude_rank": [0.6, 0.9],
        "raw_return_spread": [0.01, -0.02], "raw_magnitude_rank": [0.6, 0.9],
        "btc_variation_rank": [0.9, 0.9], "btc_realized_variation": [0.01, 0.02],
        "mstr_return": [0.01, -0.03], "coin_return": [0.0, 0.01],
        "mstr_prior_scale": [0.1, 0.1], "coin_prior_scale": [0.1, 0.1],
        "mstr_standardized_return": [0.1, -0.3], "coin_standardized_return": [0.0, 0.1],
    })
    clock = support.build_clock(frame)
    assert len(clock) == 1
    assert clock.entry_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == ["2023-07-05T20:10Z"]
    assert clock.side.tolist() == [-1]
    assert clock.split.tolist() == ["train"]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA256 == "0ef5cd7df670cf60b36f618e8bce9b85540cf7705f6460aa64360d6f6d801e28"
    assert support.SYMBOLS == ("MSTR", "COIN")
    assert support.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
