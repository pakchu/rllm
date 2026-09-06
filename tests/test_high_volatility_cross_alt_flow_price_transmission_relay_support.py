import numpy as np
import pandas as pd

from training import build_high_volatility_cross_alt_flow_price_transmission_relay_support as subject


def test_transmission_requires_four_in_one_direction():
    flow = np.array([0.1, 0.2, 0.3, 0.4, -0.2, -0.1])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.01, -0.01])
    side, breadth, score = subject.transmission(flow, returns)
    assert (side, breadth) == (1, 4)
    assert score > 0


def test_transmission_rejects_insufficient_breadth():
    flow = np.array([0.1, 0.2, 0.3, -0.4, -0.2, -0.1])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.01, 0.01])
    assert subject.transmission(flow, returns)[:2] == (0, 3)


def test_causal_rank_excludes_current_value(monkeypatch):
    monkeypatch.setitem(subject.P, "history_decisions", 3)
    monkeypatch.setitem(subject.P, "minimum_history_decisions", 2)
    ranked = subject.causal(pd.Series([1.0, 3.0, 2.0]))
    assert np.isnan(ranked.iloc[0]) and np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 0.5


def test_onset_requires_prior_valid_decision():
    state = pd.Series([True, False, True, True])
    valid = pd.Series([True, True, True, True])
    assert subject.onset(state, valid).tolist() == [False, False, True, False]
