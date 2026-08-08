import numpy as np
import pandas as pd

from training import build_russell_volatility_spread_break_relay_support as support


def test_causal_z_excludes_current_observation():
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    observed = support.causal_z(values, lookback=3, minimum=3)
    expected = (100.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=1)
    assert np.isnan(observed.iloc[:3]).all()
    assert observed.iloc[3] == expected


def test_strict_prior_midrank_excludes_current_and_caps_history():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    observed = support.strict_prior_midrank(values, lookback=2, minimum=2)
    assert np.isnan(observed.iloc[:2]).all()
    assert observed.iloc[2] == 1.0
    assert observed.iloc[3] == 1.0


def test_clock_uses_next_session_0935_new_york_and_half_open_reservation():
    decision = pd.Timestamp("2024-07-01 09:30", tz=support.NY).tz_convert("UTC")
    features = pd.DataFrame({
        "cboe_observation_date": [pd.Timestamp("2024-06-28"), pd.Timestamp("2024-06-28")],
        "next_cboe_source_date": [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-01")],
        "decision_time": [decision, decision + pd.Timedelta(hours=1)],
        "relative_volatility_shock": [0.1, -0.1], "shock_z": [2.0, -2.0],
        "vix_change": [0.1, -0.1], "vix_change_z": [2.0, -2.0],
        "btc_realized_variation": [0.1, 0.1], "btc_variation_rank": [0.9, 0.9],
    })
    clock = support.build_clock(features)
    assert len(clock) == 1
    assert pd.Timestamp(clock.iloc[0].entry_time) == decision + pd.Timedelta(minutes=5)
    assert pd.Timestamp(clock.iloc[0].exit_time) == decision + pd.Timedelta(hours=12, minutes=5)
    assert int(clock.iloc[0].side) == -1


def test_builder_is_outcome_blind_and_controls_are_nonpromotable():
    source = support.BUILDER.read_text()
    assert "funding_rates_binance" not in source
    assert "high" not in support.QUERY.lower()
    assert "low" not in support.QUERY.lower()
    assert '"advance_to_economic_outcomes": False' in source
    assert '"promotion_authorized": False' in source
