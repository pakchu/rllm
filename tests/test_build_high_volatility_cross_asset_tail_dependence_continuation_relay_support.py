import numpy as np
import pandas as pd

from training import build_high_volatility_cross_asset_tail_dependence_continuation_relay_support as support


def test_empirical_midranks_and_strict_prior_exclude_current() -> None:
    assert np.allclose(support.empirical_midranks(np.array([1.0, 2.0, 3.0])), [1/6, 1/2, 5/6])
    ranked = support.strict_prior_midrank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all() and ranked.iloc[180] == 1.0


def test_tail_dependence_detects_upper_dominance() -> None:
    rows = 181
    base = np.arange(rows, dtype=float)
    frame = pd.DataFrame({"block_start": pd.date_range("2023-01-01", periods=rows, freq="4h", tz="UTC"), "decision_time": pd.date_range("2023-01-01T04:00:00Z", periods=rows, freq="4h"), "joint_valid": True})
    frame["BTCUSDT_return"] = base
    for symbol in support.ALT_SYMBOLS: frame[f"{symbol}_return"] = base
    out = support.add_tail_dependence(frame)
    assert out.tail_asymmetry.iloc[180] == 0.0
    assert out.current_btc_tail_rank.iloc[180] > 0.99


def test_clock_side_and_timing() -> None:
    states = pd.DataFrame({"block_start": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T04:00:00Z", "2024-01-01T08:00:00Z"]), "decision_time": pd.to_datetime(["2024-01-01T04:00:00Z", "2024-01-01T08:00:00Z", "2024-01-01T12:00:00Z"]), "joint_valid": True, "tail_asymmetry": [0.2, 0.0, -0.2], "asymmetry_magnitude_rank": [0.9, 0.0, 0.9], "current_btc_tail_rank": [0.9, 0.5, 0.1], "btc_variation": 1.0, "btc_variation_rank": 0.9})
    clock = support.build_clock(states)
    assert list(clock.side) == [1, -1]
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
