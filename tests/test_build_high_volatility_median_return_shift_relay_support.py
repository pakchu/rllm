import numpy as np
import pandas as pd

from training import build_high_volatility_median_return_shift_relay_support as s


def test_shift_statistics_detects_positive_location_shift():
    open_ = np.repeat(100.0, 480)
    returns = np.r_[np.linspace(-0.002, 0.0, 240), np.linspace(0.001, 0.003, 240)]
    close = open_ * np.exp(returns)
    block = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
        }
    )

    median_shift, mean_shift, variation = s.shift_statistics(block)

    assert median_shift > 0
    assert mean_shift > 0
    assert variation > 0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "median_shift": [-0.1, 0.3, 0.4, 0.2, -0.5, -0.4],
            "shift_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "mean_shift": [0.1, -0.3, -0.4, -0.2, 0.5, 0.4],
            "mean_shift_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "variation_rank": [0.8] * 6,
            "feature_available_time": pd.date_range(
                "2024-01-01", periods=6, freq="8h", tz="UTC"
            ),
        }
    )


def test_primary_and_controls():
    active, side, _ = s.active(panel())
    assert active.tolist() == [False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1]

    without_variation = panel()
    without_variation.loc[1, "variation_rank"] = 0.4
    assert s.active(without_variation, "no_variation_gate")[0].iloc[1]

    mean_active, mean_side, _ = s.active(panel(), "mean_return_shift")
    assert mean_active.tolist() == [False, True, False, False, True, False]
    assert mean_side[mean_active].tolist() == [-1, 1]


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0
