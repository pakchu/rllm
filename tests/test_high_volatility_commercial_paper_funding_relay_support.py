import numpy as np
import pandas as pd

from training import build_high_volatility_commercial_paper_funding_relay_support as support


def test_rank_excludes_current_and_uses_required_history():
    ranked = support.strict_prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def state(**changes):
    row = {
        "reference_date": pd.Timestamp("2023-07-05T00:00:00Z"),
        "first_vintage_date": pd.Timestamp("2023-07-06T00:00:00Z"),
        "revised_vintage_date": pd.Timestamp("2023-07-13T00:00:00Z"),
        "decision_time": pd.Timestamp("2023-07-14T05:05:00Z"),
        "stock_change": 0.1, "stock_first_change": 0.2,
        "btc_variation": 0.2, "btc_variation_rank": 0.8,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_primary_buys_funding_expansion_and_clock():
    states = state()
    assert support.build_clock(states).side.tolist() == [1]
    assert support.build_clock(states).entry_time.iloc[0] == pd.Timestamp("2023-07-14T05:10:00Z")
    assert support.build_clock(states, "direction_flip").side.tolist() == [-1]
    assert support.build_clock(states, "same_clock_forced_long").side.tolist() == [1]


def test_variation_gate_is_removed_only_by_control():
    states = state(btc_variation_rank=0.2)
    assert support.build_clock(states).empty
    assert support.build_clock(states, "no_variation_gate").side.tolist() == [1]


def test_stale_control_uses_prior_change_but_current_variation():
    prior = state(reference_date=pd.Timestamp("2023-06-28T00:00:00Z"), first_vintage_date=pd.Timestamp("2023-06-29T00:00:00Z"), revised_vintage_date=pd.Timestamp("2023-07-06T00:00:00Z"), decision_time=pd.Timestamp("2023-07-07T05:05:00Z"))
    current = state(stock_change=-0.2)
    states = pd.concat([prior, current], ignore_index=True)
    clock = support.build_clock(states, "one_release_stale_stock")
    assert clock.reference_date.tolist() == [pd.Timestamp("2023-06-28T00:00:00Z")]
    assert clock.side.tolist() == [1]


def test_first_print_control_uses_first_change_only():
    states = state(stock_change=10.0, stock_first_change=-5.0)
    assert support.build_clock(states).side.tolist() == [1]
    assert support.build_clock(states, "current_first_print_stock").side.tolist() == [-1]
