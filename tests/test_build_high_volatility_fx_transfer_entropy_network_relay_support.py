import numpy as np
import pandas as pd

from training import build_high_volatility_fx_transfer_entropy_network_relay_support as support


def test_transfer_entropy_detects_known_binary_source() -> None:
    rng = np.random.default_rng(31)
    source = pd.Series(rng.choice([-1, 1], size=500), dtype=float)
    target = pd.Series(np.r_[1, source.to_numpy()[:-1]], dtype=float)
    forward, samples, minimum_cell = support.transfer_entropy(source, target)
    reverse, _, _ = support.transfer_entropy(target, source)
    assert samples == 499
    assert minimum_cell >= 20
    assert forward > 0.5
    assert forward > reverse


def test_transfer_entropy_network_finds_unique_nonlinear_source() -> None:
    rng = np.random.default_rng(7)
    leader = rng.choice([-1, 1], size=700)
    frame = {"EURUSD": leader}
    for index, symbol in enumerate(support.SYMBOLS[1:], 1):
        follower = np.r_[1, leader[:-1]].copy()
        flips = rng.random(700) < (0.03 + 0.01 * index)
        follower[flips] *= -1
        frame[symbol] = follower
    source, strength, breadth, _ = support.transfer_entropy_network(pd.DataFrame(frame, dtype=float))
    assert source == "EURUSD"
    assert strength > 0
    assert breadth == 5


def test_strict_prior_midrank_excludes_current() -> None:
    ranks = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert np.isnan(ranks.iloc[0])
    assert np.isnan(ranks.iloc[1])
    assert ranks.iloc[2] == 1.0


def test_onset_compares_previous_source_valid_session() -> None:
    source_valid = pd.Series([True, False, True, True, True])
    eligible = pd.Series([False, False, True, True, False])
    onset = support.onset_after_previous_source_valid(source_valid, eligible)
    assert onset.tolist() == [False, False, True, False, False]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA == "6053ab00cf402534e49a26ed7c553bc8e1f8d6ae32a7adf6d2c5e9d1d3ee29c1"
    assert support.CONTROLS == (
        "no_source_strength_tail",
        "no_variation_gate",
        "no_breadth_gate",
        "linear_lag_network",
        "one_session_stale_network",
        "direction_flip",
        "same_clock_forced_long",
    )
    assert "bars_polygon" in support.QUERY
    assert "bars_binance" not in support.QUERY
