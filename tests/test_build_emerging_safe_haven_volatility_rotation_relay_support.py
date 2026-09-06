import numpy as np
import pandas as pd

from training import build_emerging_safe_haven_volatility_rotation_relay_support as support


def test_esvrr_midrank_excludes_current():
    rank = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0


def test_esvrr_primary_is_tail_high_variation_and_fades_change():
    frame = pd.DataFrame({
        "relative_volatility_change": [0.1, -0.1, 0.1],
        "absolute_change_rank": [0.8, 0.8, 0.6],
        "btc_variation_rank": [0.8, 0.5, 0.8],
        "vxeem_minus_gvz_raw": [1.0, 2.0, 3.0],
    })
    assert support.signal(frame, "primary").tolist() == [-1, 0, 0]
    assert support.signal(frame, "no_btc_variation_gate").tolist() == [-1, 1, 0]
    assert support.signal(frame, "direction_flip").tolist() == [1, 0, 0]


def test_esvrr_clock_is_0935_new_york_with_twelve_hour_hold():
    decision = pd.Timestamp("2024-07-01 09:30", tz=support.NY).tz_convert("UTC")
    frame = pd.DataFrame({
        "cboe_observation_date": [pd.Timestamp("2024-06-28")],
        "next_cboe_source_date": [pd.Timestamp("2024-07-01")],
        "decision_time": [decision], "relative_volatility": [0.1],
        "relative_volatility_change": [0.1], "absolute_change_rank": [0.8],
        "vxeem_minus_gvz_raw": [1.0], "btc_realized_variation": [0.1],
        "btc_variation_rank": [0.8],
    })
    clock = support.build_clock(frame)
    assert len(clock) == 1
    assert pd.Timestamp(clock.iloc[0].entry_time) == decision + pd.Timedelta(minutes=5)
    assert pd.Timestamp(clock.iloc[0].exit_time) == decision + pd.Timedelta(hours=12, minutes=5)


def test_esvrr_builder_is_outcome_blind():
    source = support.BUILDER.read_text()
    assert "funding_rates_binance" not in source
    assert '"advance_to_economic_outcomes": False' in source
    assert '"promotion_authorized": False' in source
