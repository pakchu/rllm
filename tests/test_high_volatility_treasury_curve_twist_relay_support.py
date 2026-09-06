import numpy as np
import pandas as pd

from training import build_high_volatility_treasury_curve_twist_relay_support as support


def test_rank_excludes_current_and_uses_required_history():
    ranked = support.strict_prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def state(**changes):
    row = {
        "source_day": pd.Timestamp("2023-07-03T00:00:00Z"),
        "decision_time": pd.Timestamp("2023-07-04T12:00:00Z"),
        "transition_valid": True,
        "twist_valid": True,
        "delta_2y": -0.05,
        "delta_10y": 0.03,
        "twist": 0.08,
        "btc_variation": 0.2,
        "btc_variation_rank": 0.8,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_follows_curve_twist_and_controls_are_diagnostic():
    states = state()
    assert support.build_clock(states).side.tolist() == [1]
    assert support.build_clock(states).entry_time.iloc[0] == pd.Timestamp("2023-07-04T12:05:00Z")
    assert support.build_clock(states).exit_time.iloc[0] == pd.Timestamp("2023-07-05T12:05:00Z")
    assert support.build_clock(states, "direction_flip").side.tolist() == [-1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]


def test_variation_gate_is_removed_only_by_control():
    states = state(btc_variation_rank=0.2)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_variation_gate").side.tolist() == [1]


def test_single_tenor_controls_do_not_change_primary_clock_geometry():
    states = state()
    assert support.build_clock(states, "two_year_only").side.tolist() == [1]
    assert support.build_clock(states, "ten_year_only").side.tolist() == [1]


def test_stale_control_uses_prior_twist_with_current_variation():
    prior = state(source_day=pd.Timestamp("2023-06-30T00:00:00Z"), decision_time=pd.Timestamp("2023-07-01T12:00:00Z"))
    current = state(delta_2y=0.04, delta_10y=-0.02, twist=-0.06)
    states = pd.concat([prior, current], ignore_index=True)
    clock = support.build_clock(states, "one_observation_stale_twist")
    assert clock.source_day.tolist() == [pd.Timestamp("2023-06-30T00:00:00Z")]
    assert clock.side.tolist() == [1]
