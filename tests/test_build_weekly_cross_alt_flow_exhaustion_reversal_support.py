import numpy as np
import pandas as pd

from training import build_weekly_cross_alt_flow_exhaustion_reversal_support as support


def test_strict_prior_rank_excludes_current_and_caps_history():
    rank = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0, 2.0]), lookback=2, minimum=2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25


def test_primary_and_frozen_controls():
    frame = pd.DataFrame({
        "session_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "decision_time": pd.date_range("2024-01-01T08:00:00Z", periods=3, freq="1D"),
        "btc_flow_acceleration": [0.02, -0.02, 0.02], "confirming_alts": [5, 6, 3],
        "alt_majority_sign": [-1, 1, -1],
        "btc_realized_variation": [0.1, 0.1, 0.1], "variation_rank": [0.7, 0.6, 0.7],
    })
    active, side, _ = support.conditions(frame)
    assert active.tolist() == [True, False, False]
    assert side.tolist() == [1, -1, 1]
    assert support.conditions(frame, "no_volatility_gate")[0].tolist() == [True, True, False]
    assert support.conditions(frame, "three_of_six_breadth")[0].tolist() == [True, False, True]
    assert support.conditions(frame, "btc_weekly_confirmation")[0].tolist() == [False, False, False]
    assert support.conditions(frame, "direction_flip")[1].tolist() == [-1, 1, -1]


def test_clock_uses_five_minute_delay_holds_24h_and_reserves_half_open():
    frame = pd.DataFrame({
        "session_date": ["2024-01-01", "2024-01-01"],
        "decision_time": [pd.Timestamp("2024-01-01T08:00:00Z"), pd.Timestamp("2024-01-01T10:00:00Z")],
        "btc_flow_acceleration": [-0.02, 0.02], "confirming_alts": [5, 6],
        "alt_majority_sign": [-1, 1],
        "btc_realized_variation": [0.1, 0.1], "variation_rank": [0.7, 0.7],
    })
    clock = support.build_clock(frame)
    assert len(clock) == 1
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-01-01T08:05:00Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-01-02T08:05:00Z")


def test_builder_is_bound_and_outcomes_are_sealed():
    source = support.BUILDER.read_text()
    assert support.PREREG_SHA == "929f0bc729ad56eab4b63af9a495d270c1d4c315a517bd155d0e14f68966698b"
    assert "FROM bars_binance" in source
    assert "funding_rates_binance" not in source
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
    assert '"promotion_authorized": False' in source
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
