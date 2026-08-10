import numpy as np
import pandas as pd

from training import build_high_volatility_slow_donchian_breakout_relay_support as s


def test_block_statistics_uses_slow_auction_extrema_and_variation():
    returns = np.linspace(-0.002, 0.003, 96)
    block = pd.DataFrame(
        {
            "open": np.repeat(100.0, 480),
            "high": np.repeat(102.0, 480),
            "low": np.repeat(98.0, 480),
            "close": np.repeat(100 * np.exp(returns), 5),
        }
    )
    high, low, close, variation = s.block_statistics(block)
    assert high == 102.0
    assert low == 98.0
    assert close == block.close.iloc[-1]
    assert variation > 0


def test_breakout_side_requires_exclusive_close_or_touch():
    upper = pd.Series([10.0, 10.0, 10.0])
    lower = pd.Series([5.0, 5.0, 5.0])
    high = pd.Series([11.0, 9.0, 11.0])
    low = pd.Series([6.0, 4.0, 4.0])
    close = pd.Series([10.5, 4.5, 7.0])
    assert s.breakout_side(upper, lower, high, low, close).tolist() == [1, -1, 0]
    assert s.breakout_side(upper, lower, high, low, close, True).tolist() == [1, -1, 0]


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "block_high": [10.0] * 6,
            "block_low": [5.0] * 6,
            "block_close": [7.0] * 6,
            "channel_high_30": [9.0] * 6,
            "channel_low_30": [6.0] * 6,
            "channel_high_15": [8.0] * 6,
            "channel_low_15": [6.0] * 6,
            "close_breakout_side": [0, 1, 1, 0, -1, -1],
            "touch_breakout_side": [0, 1, 1, 0, -1, -1],
            "realized_variation": [0.1] * 6,
            "variation_rank": [0.8] * 6,
            "feature_available_time": pd.date_range(
                "2024-01-01", periods=6, freq="8h", tz="UTC"
            ),
        }
    )


def test_primary_onset_and_side():
    active, side, _ = s.active(panel(), "primary")
    assert active.tolist() == [False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1]


def test_controls_are_diagnostic():
    frame = panel()
    frame.loc[1, "variation_rank"] = 0.4
    assert s.active(frame, "no_variation_gate")[0].iloc[1]
    active, side, _ = s.active(panel(), "direction_flip")
    assert side[active].tolist() == [-1, 1]


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0
