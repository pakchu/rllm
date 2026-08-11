import numpy as np
import pandas as pd

from training import build_high_volatility_realized_leverage_cross_moment_relay_support as b


def _five(returns: np.ndarray) -> pd.DataFrame:
    opens = np.full(len(returns), 100.0)
    closes = opens * np.exp(returns)
    return pd.DataFrame({"bar_time": pd.date_range("2023-01-01", periods=len(returns), freq="5min", tz="UTC"), "valid": True, "open": opens, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes), "close": closes, "return": returns})


def test_negative_returns_followed_by_variance_make_negative_cross_moment():
    pattern = np.array([-0.02, 0.03, -0.01, 0.025] * 72)
    states = b.derive_moments(_five(pattern))
    finite = states[np.isfinite(states.leverage_cross_moment)]
    assert len(finite) == 1
    assert finite.leverage_cross_moment.iloc[0] < 0
    assert finite.btc_realized_variation.iloc[0] > 0


def test_strict_prior_midrank_excludes_current():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    valid = pd.Series([True] * 4)
    rank = b.strict_prior_midrank(values, valid, lookback=3, minimum=3)
    assert rank.iloc[:3].isna().all()
    assert rank.iloc[3] == 1.0


def test_clock_reserves_half_open_twelve_hours():
    frame = pd.DataFrame({"decision_time": pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T04:00:00Z", "2023-07-01T12:00:00Z"]), "leverage_cross_moment": [-0.2, 0.3, 0.4], "contemporaneous_moment": [-0.1, 0.2, 0.3], "magnitude_rank": [0.9, 0.9, 0.9], "contemporaneous_magnitude_rank": [0.9, 0.9, 0.9], "btc_realized_variation": [0.1, 0.2, 0.3], "btc_variation_rank": [0.9, 0.9, 0.9]})
    clock = b.build_clock(frame)
    assert len(clock) == 2
    assert clock.side.tolist() == [-1, 1]
    assert clock.entry_time.iloc[1] == pd.Timestamp("2023-07-01T12:05:00Z")
