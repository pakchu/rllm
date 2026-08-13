import numpy as np
import pandas as pd

from training import build_high_volatility_premium_implied_funding_surprise_reversal_support as support


def test_strict_prior_midrank_excludes_current() -> None:
    values = pd.Series(np.arange(181, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:180].isna().all()
    assert ranks.iloc[180] == 1.0


def test_published_clamp_map() -> None:
    assert support.implied_funding(0.0) == 0.0001
    assert support.implied_funding(0.001) == 0.0005
    assert support.implied_funding(-0.001) == -0.0005


def test_surprise_maps_to_contrarian_side_and_onset() -> None:
    decisions = pd.date_range("2023-07-01T00:00:00Z", periods=4, freq="8h")
    panel = pd.DataFrame({
        "decision_time": decisions,
        "funding_rate": [0.00001, 0.001, 0.00001, -0.001],
        "premium_average": [0.0] * 4,
        "implied_funding_proxy": [0.0] * 4,
        "funding_surprise": [0.00001, 0.001, 0.00001, -0.001],
        "unweighted_funding_surprise": [0.00001, 0.001, 0.00001, -0.001],
        "surprise_rank": [0.1, 0.8, 0.1, 0.8],
        "unweighted_surprise_rank": [0.1, 0.8, 0.1, 0.8],
        "btc_variation": [1.0] * 4,
        "variation_rank": [0.8] * 4,
        "source_valid": [True] * 4,
    })
    clock = support.candidate_clock(panel)
    assert clock.side.tolist() == [-1, 1]
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=8)).all()
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
