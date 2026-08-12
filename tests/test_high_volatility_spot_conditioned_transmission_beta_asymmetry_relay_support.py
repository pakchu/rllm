import numpy as np
import pandas as pd

from training import build_high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_support as support


def test_midrank_excludes_current():
    ranked = support.strict_prior_midrank(pd.Series(np.arange(181, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def test_transmission_metrics_are_conditioned_on_spot_sign():
    spot_returns = np.r_[np.full(240, 0.001), np.full(240, -0.001)]
    perp_returns = np.r_[2.0 * spot_returns[:240], 0.5 * spot_returns[240:]]
    spot = np.exp(np.r_[0.0, np.cumsum(spot_returns)])
    perp = np.exp(np.r_[0.0, np.cumsum(perp_returns)])
    metrics = support.transmission_metrics(spot, perp)
    assert metrics["source_valid"] is True
    assert np.isclose(metrics["up_spot_beta"], 2.0)
    assert np.isclose(metrics["down_spot_beta"], 0.5)
    assert np.isclose(metrics["transmission_asymmetry"], np.log(0.25))


def _frame(**updates):
    rows = []
    for index in range(2):
        row = {
            "decision_time": pd.Timestamp("2023-07-01T00:00:00Z") + pd.Timedelta(hours=8 * index),
            "source_valid": True, "up_spot_beta": 1.0, "down_spot_beta": 2.0,
            "transmission_asymmetry": np.log(2.0), "asymmetry_rank": 0.9,
            "positive_spot_minutes": 240, "negative_spot_minutes": 240,
            "full_variation": 0.1, "variation_rank": 0.8,
        }
        row.update(updates)
        rows.append(row)
    return pd.DataFrame(rows)


def test_primary_uses_onset_and_shorts_downside_amplification():
    frame = _frame()
    frame.loc[0, "asymmetry_rank"] = 0.2
    primary = support.clock(frame)
    assert primary.side.tolist() == [-1]
    assert primary.decision_time.tolist() == [pd.Timestamp("2023-07-01T08:00:00Z")]
    assert len(support.clock(frame, "no_onset")) == 1


def test_named_controls_do_not_change_primary_rule():
    frame = _frame(asymmetry_rank=0.2, variation_rank=0.2)
    assert support.clock(frame).empty
    assert len(support.clock(frame, "no_asymmetry_tail")) == 0
    variation_only = _frame(asymmetry_rank=0.9, variation_rank=0.2)
    variation_only.loc[0, "asymmetry_rank"] = 0.2
    assert support.clock(variation_only).empty
    assert support.clock(variation_only, "no_variation_gate").side.tolist() == [-1]
