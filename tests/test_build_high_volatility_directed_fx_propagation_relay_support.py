import numpy as np
import pandas as pd

from training import build_high_volatility_directed_fx_propagation_relay_support as support


def test_directed_network_finds_unique_source():
    rng = np.random.default_rng(3)
    leader = rng.normal(size=500)
    series = {"EURUSD": pd.Series(leader)}
    for index, symbol in enumerate(support.SYMBOLS[1:], 1):
        follower = np.r_[0.0, leader[:-1]] + rng.normal(scale=0.05 + index * 0.002, size=500)
        series[symbol] = pd.Series(follower)
    source, strength, breadth, _ = support.directed_network(series)
    assert source == "EURUSD"
    assert strength > 0.0 and breadth == 5


def test_strict_prior_midrank_excludes_current():
    ranks = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert np.isnan(ranks.iloc[0]) and np.isnan(ranks.iloc[1])
    assert ranks.iloc[2] == 1.0
