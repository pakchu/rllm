import numpy as np
import pandas as pd

from training import build_high_volatility_stock_bitcoin_coskewness_relay_support as b


def test_strict_prior_zscore_excludes_current_and_uses_population_scale():
    values = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
    scores = b.strict_prior_zscore(values, lookback=3, minimum=3)
    assert scores.iloc[:3].isna().all()
    expected = (100.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=0)
    assert scores.iloc[3] == expected
    changed = values.copy()
    changed.iloc[4] = 200.0
    assert b.strict_prior_zscore(changed, lookback=3, minimum=3).iloc[3] == scores.iloc[3]


def test_strict_prior_midrank_excludes_current():
    values = pd.Series([1.0, 2.0, 3.0])
    ranks = b.strict_prior_midrank(values, lookback=2, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == 1.0


def test_clock_uses_actual_early_close_and_frozen_delays():
    frame = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2023-07-03")],
            "cash_close_time": [pd.Timestamp("2023-07-03T17:00:00Z")],
            "spy_daily_return": [0.01], "btc_daily_return": [0.02], "coskewness": [0.4],
            "coskewness_z": [1.5], "btc_skewness": [0.2], "btc_skewness_z": [1.1],
            "btc_realized_variation": [0.03], "btc_variation_rank": [0.9],
        }
    )
    clock = b.build_clock(frame)
    assert clock.entry_time.iloc[0] == pd.Timestamp("2023-07-03T17:10:00Z")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2023-07-04T17:10:00Z")
    assert clock.side.iloc[0] == -1


def test_rolling_coskewness_uses_fixed_completed_window():
    btc = pd.Series(np.arange(1.0, 7.0))
    spy = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    actual = b.rolling_coskewness(btc, spy, window=5)
    assert actual.iloc[:4].isna().all()
    x, y = btc.iloc[:5].to_numpy(), spy.iloc[:5].to_numpy()
    expected = np.mean((x-x.mean())*(y-y.mean())**2)/(x.std(ddof=0)*y.var(ddof=0))
    assert actual.iloc[4] == expected
