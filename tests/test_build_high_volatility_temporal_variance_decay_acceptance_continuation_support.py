import numpy as np
import pandas as pd

from training import build_high_volatility_temporal_variance_decay_acceptance_continuation_support as s


def test_prior_rank_excludes_current(monkeypatch):
    monkeypatch.setitem(s.POLICY, "history_cycles", 3)
    monkeypatch.setitem(s.POLICY, "minimum_history_cycles", 2)
    got = s.prior_rank(pd.Series([1.0, 2.0, 3.0, 2.0]))
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert got.iloc[2] == 1.0 and got.iloc[3] == 0.5


def test_clock_follows_accepted_direction(monkeypatch):
    monkeypatch.setattr(s, "stage_for", lambda *_: "train")
    d = pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T16:00:00Z"])
    panel = pd.DataFrame({
        "decision_time": d, "feature_available_time": d, "onset": [True, True],
        "first_half_return": [0.02, -0.02], "second_half_return": [0.01, -0.01],
        "first_half_qv": [0.1, 0.1], "second_half_qv": [0.05, 0.05],
        "first_half_efficiency": [0.1, 0.1], "second_half_efficiency": [0.2, 0.2],
        "realized_variation": [0.3, 0.3], "variation_rank": [0.8, 0.9],
    })
    assert s.build_clock(panel).side.tolist() == [1, -1]


def test_preregistration_is_hash_bound():
    assert s.sha256(s.prereg.DEFAULT_OUTPUT) == s.PREREG_SHA
    s.prereg.validate(s.REGISTRATION)
