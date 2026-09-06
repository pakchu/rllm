import numpy as np
import pandas as pd

from training import build_high_volatility_nfci_weekend_relay_support as support


def test_rank_excludes_current_and_uses_required_history():
    ranked = support.strict_prior_rank(pd.Series(np.arange(105, dtype=float)))
    assert ranked.iloc[:104].isna().all()
    assert ranked.iloc[104] == 1.0


def state(**changes):
    row = {
        "reference_date": pd.Timestamp("2023-07-07T00:00:00Z"),
        "vintage_date": pd.Timestamp("2023-07-14T00:00:00Z"),
        "decision_time": pd.Timestamp("2023-07-14T00:00:00Z"),
        "nfci_change": 0.1,
        "btc_variation": 0.2,
        "btc_variation_rank": 0.8,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_fades_tightening_and_controls_are_diagnostic():
    states = state()
    assert support.build_clock(states).side.tolist() == [-1]
    assert support.build_clock(states).entry_time.iloc[0] == pd.Timestamp("2023-07-14T00:05:00Z")
    assert support.build_clock(states, "direction_flip").side.tolist() == [1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]


def test_variation_gate_is_removed_only_by_control():
    states = state(btc_variation_rank=0.2)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_variation_gate").side.tolist() == [-1]


def test_stale_control_uses_prior_change_but_current_variation():
    prior = state(
        reference_date=pd.Timestamp("2023-06-30T00:00:00Z"),
        vintage_date=pd.Timestamp("2023-07-07T00:00:00Z"),
        decision_time=pd.Timestamp("2023-07-07T00:00:00Z"),
    )
    current = state(nfci_change=-0.2)
    states = pd.concat([prior, current], ignore_index=True)
    clock = support.build_clock(states, "one_week_stale_nfci")
    assert clock.reference_date.tolist() == [pd.Timestamp("2023-06-30T00:00:00Z")]
    assert clock.side.tolist() == [-1]
