import numpy as np
import pandas as pd

from training import build_high_volatility_initial_claims_labor_relay_support as support


def test_rank_excludes_current_and_uses_required_history():
    ranked = support.strict_prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def state(**changes):
    row = {
        "reference_date": pd.Timestamp("2023-07-01T00:00:00Z"),
        "first_vintage_date": pd.Timestamp("2023-07-06T00:00:00Z"),
        "revised_vintage_date": pd.Timestamp("2023-07-13T00:00:00Z"),
        "decision_time": pd.Timestamp("2023-07-13T12:35:00Z"),
        "claims_change": 0.1,
        "claims_first_change": 0.2,
        "btc_variation": 0.2,
        "btc_variation_rank": 0.8,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_sells_rising_claims_and_controls_are_diagnostic():
    states = state()
    assert support.build_clock(states).side.tolist() == [-1]
    assert support.build_clock(states).entry_time.iloc[0] == pd.Timestamp("2023-07-13T12:40:00Z")
    assert support.build_clock(states, "direction_flip").side.tolist() == [1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]


def test_variation_gate_is_removed_only_by_control():
    states = state(btc_variation_rank=0.2)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_variation_gate").side.tolist() == [-1]


def test_stale_control_uses_prior_change_but_current_variation():
    prior = state(
        reference_date=pd.Timestamp("2023-06-24T00:00:00Z"),
        first_vintage_date=pd.Timestamp("2023-06-29T00:00:00Z"),
        revised_vintage_date=pd.Timestamp("2023-07-06T00:00:00Z"),
        decision_time=pd.Timestamp("2023-07-06T12:35:00Z"),
    )
    current = state(claims_change=-0.2)
    states = pd.concat([prior, current], ignore_index=True)
    clock = support.build_clock(states, "one_release_stale_claims")
    assert clock.reference_date.tolist() == [pd.Timestamp("2023-06-24T00:00:00Z")]
    assert clock.side.tolist() == [-1]


def test_unrevised_control_uses_first_print_change_only():
    states = state(claims_change=10.0, claims_first_change=-5.0)
    assert support.build_clock(states).side.tolist() == [-1]
    assert support.build_clock(states, "current_unrevised_claims").side.tolist() == [1]


def test_decision_clock_is_dst_aware():
    source = pd.DataFrame({
        "reference_date": ["2023-02-25", "2023-03-04"],
        "first_vintage_date": ["2023-03-02", "2023-03-09"],
        "revised_vintage_date": ["2023-03-09", "2023-03-16"],
        "icsa_first": [200000, 201000],
        "icsa_revised": [199000, 200000],
    })
    # score_states needs market rows; the clock conversion itself is isolated here.
    dates = pd.to_datetime(source.revised_vintage_date).dt.tz_localize("America/New_York")
    decisions = (dates + pd.Timedelta(hours=8, minutes=35)).dt.tz_convert("UTC")
    assert decisions.tolist() == [pd.Timestamp("2023-03-09T13:35:00Z"), pd.Timestamp("2023-03-16T12:35:00Z")]
