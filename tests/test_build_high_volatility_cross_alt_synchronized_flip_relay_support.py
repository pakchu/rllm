import numpy as np
import pandas as pd

from training import build_high_volatility_cross_alt_synchronized_flip_relay_support as support


def _blocks(first_signs, second_signs):
    times = pd.date_range("2026-01-01T00:00:00Z", periods=480, freq="1min")
    rows = []
    for symbol, first_sign, second_sign in zip(support.ALTS, first_signs, second_signs):
        first = np.exp(np.linspace(0.0, first_sign * 0.01, 240))
        second = np.exp(np.linspace(0.0, second_sign * 0.01, 240))
        for time, value in zip(times, np.r_[first, second]):
            rows.append((time, symbol, value, value))
    alt = pd.DataFrame(rows, columns=["ts", "symbol", "open", "close"]).set_index(["ts", "symbol"])
    btc = pd.DataFrame({"open": np.ones(1440), "close": np.exp(np.linspace(0.0, 0.02, 1440))})
    return alt, btc


def test_flip_statistics_detects_broad_majority_reset():
    alt, btc = _blocks([1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, 1, 1])
    first_positive, second_positive, first_majority, second_majority, flip_count, majority_flip, variation = support.flip_statistics(alt, btc)
    assert (first_positive, second_positive) == (4, 2)
    assert (first_majority, second_majority) == (1, -1)
    assert flip_count == 6
    assert majority_flip is True
    assert variation > 0


def test_active_primary_requires_four_symbol_flips_and_variation():
    panel = pd.DataFrame(
        {
            "source_valid": [True, True],
            "majority_flip": [True, True],
            "symbol_flip_count": [3, 4],
            "variation_rank": [0.9, 0.9],
            "second_majority": [-1, 1],
            "feature_available_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True
            ),
        }
    )
    active, side, _ = support.active(panel)
    assert active.tolist() == [False, True]
    assert side.tolist() == [-1, 1]
