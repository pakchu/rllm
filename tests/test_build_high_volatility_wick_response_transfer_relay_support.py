import numpy as np
import pandas as pd

from training import build_high_volatility_wick_response_transfer_relay_support as s


def test_wick_statistics_uses_separated_response_and_pressure_windows():
    wick = np.linspace(-0.03, 0.03, 96)
    returns = np.zeros(96)
    returns[1:84] = wick[:83]
    open_ = np.repeat(100.0, 96)
    close = open_ * np.exp(returns)
    upper = np.maximum(-wick, 0.0)
    lower = np.maximum(wick, 0.0)
    high = np.maximum(open_, close) * np.exp(upper)
    low = np.minimum(open_, close) * np.exp(-lower)
    block = pd.DataFrame(
        {
            "open": np.repeat(open_, 5),
            "high": np.repeat(high, 5),
            "low": np.repeat(low, 5),
            "close": np.repeat(close, 5),
        }
    )
    response, pressure, variation, signal = s.wick_statistics(block)
    assert response > 0.99
    assert pressure > 0
    assert variation > 0
    assert signal > 0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "wick_response": [0.1, 0.3, 0.4, -0.2, -0.5, 0.4],
            "response_rank": [0.5, 0.85, 0.9, 0.4, 0.85, 0.9],
            "final_hour_pressure": [0.2, -0.1, 0.2, -0.3, 0.4, 0.2],
            "realized_variation": [1.0] * 6,
            "variation_rank": [0.8] * 6,
            "transfer_signal": [0.02, -0.03, 0.08, 0.06, -0.2, 0.08],
            "feature_available_time": pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
        }
    )


def test_primary_onset_and_transfer_side():
    active, side, _ = s.active(panel(), "primary")
    assert active.tolist() == [False, True, False, False, True, False]
    assert side[active].tolist() == [-1, -1]


def test_controls_are_diagnostic():
    frame = panel(); frame.loc[1, "variation_rank"] = 0.4
    assert s.active(frame, "no_variation_gate")[0].iloc[1]
    active, side, _ = s.active(panel(), "late_wick_pressure_only")
    assert side[active].tolist() == [-1, 1]
    assert s.active(panel(), "forced_long")[1].eq(1).all()


def test_prior_rank_excludes_current():
    original = s.P
    try:
        s.P = {**original, "history_hours": 10, "minimum_history_hours": 3}
        ranks = s.prior_rank(pd.Series([1.0, 2.0, 3.0, 4.0]))
        assert np.isnan(ranks.iloc[2])
        assert ranks.iloc[3] == 1.0
    finally:
        s.P = original
