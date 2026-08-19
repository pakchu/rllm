import numpy as np
import pandas as pd
from types import SimpleNamespace

from training import search_medium_frequency_volatility_alpha as alpha


def market(periods=1200):
    dates = pd.date_range("2020-01-01", periods=periods, freq="5min")
    price = np.arange(periods, dtype=float) + 100
    return pd.DataFrame({"date": dates, "open": price, "high": price + 1, "low": price - 1, "close": price + .5})


def test_causal_clock_and_next_open_features():
    frame = market()
    mask = alpha.decision_mask(frame.date)
    assert frame.loc[mask, "date"].dt.hour.tolist() == [0, 8, 16, 0, 8, 16, 0, 8, 16, 0, 8, 16, 0]
    features = alpha.build_features(frame)
    pos = 864
    before = features.iloc[pos].copy()
    frame.loc[pos, ["open", "high", "low", "close"]] = 1e9
    pd.testing.assert_series_equal(before, alpha.build_features(frame).iloc[pos])


def test_thresholds_are_train_only():
    dates = pd.date_range("2020-01-01", "2024-01-02", freq="8h")
    features = pd.DataFrame({"realized_vol_24h": np.arange(len(dates), dtype=float)})
    first = alpha.fit_thresholds(features, pd.Series(dates))
    features.loc[dates >= "2023-01-01", "realized_vol_24h"] = 1e12
    assert alpha.fit_thresholds(features, pd.Series(dates)) == first


def test_frequency_gate_requires_three_per_week_each_window():
    good = {name: {"avg_trades_per_week": 3.0} for name in alpha.SELECTION_WINDOWS}
    assert alpha.frequency_passes(good)
    good["2022"]["avg_trades_per_week"] = 2.99
    assert not alpha.frequency_passes(good)


def test_eval_source_cannot_rerank():
    source = open(alpha.__file__, encoding="utf-8").read().split("def evaluate", 1)[1].split("def main", 1)[0]
    assert "candidate_grid(" not in source
    assert "rows.sort" not in source
    assert len(alpha.candidate_grid()) == 216


def test_schedule_skips_overlapping_eight_hour_positions():
    dates = pd.date_range("2024-01-01", periods=300, freq="5min")

    class Engine:
        market = pd.DataFrame({"date": dates})

        def trade_at(self, signal, side, hold, tp, stop):
            return SimpleNamespace(exit_position=signal + hold)

    sides = np.zeros(len(dates), dtype=np.int8)
    sides[[0, 96, 192]] = 1
    trades = alpha._schedule(
        Engine(), sides, "2024-01-01", "2024-01-02 01:00", {"hold_hours": 8, "tp": None}
    )
    assert len(trades) == 2
