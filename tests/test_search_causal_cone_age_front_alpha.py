import numpy as np
import pandas as pd

from training import search_causal_cone_age_front_alpha as front


def _market(n: int = 2600):
    dates = pd.Series(pd.date_range("2020-10-01", periods=n, freq="5min"))
    close = np.exp(4.0 + np.cumsum(0.001 * np.sin(np.arange(n) / 11.0)))
    return pd.DataFrame({"close": close}), dates


def test_weighted_quantile_uses_weight_mass():
    value = front.weighted_quantile(
        np.array([0.1, 0.5, 0.9]), np.array([1.0, 1.0, 8.0]), 0.8
    )
    assert value == 0.9


def test_breach_age_profile_tracks_dominant_outward_mass():
    result = front.breach_age_profile(
        np.array([0.0, 2.5, 3.0, 5.0]), np.array([0.1, 0.3, 0.6, 0.9])
    )
    assert result["side"] == 1.0
    assert result["upper_mass"] > 0.0
    assert result["upper_front"] == 0.9
    assert result["upper_center"] > 0.5
    assert result["lower_mass"] == 0.0


def test_reversing_age_order_changes_age_geometry_not_mass():
    z = np.array([0.0, 2.5, 3.0, 5.0])
    age = np.array([0.1, 0.3, 0.6, 0.9])
    ordered = front.breach_age_profile(z, age)
    reversed_age = front.breach_age_profile(z, age[::-1])
    assert ordered["upper_mass"] == reversed_age["upper_mass"]
    assert ordered["upper_center"] != reversed_age["upper_center"]


def test_age_front_state_is_prefix_independent(monkeypatch):
    market, dates = _market()
    monkeypatch.setattr(front, "ANCHOR_HORIZON", 144)
    monkeypatch.setattr(front, "prior_volatility", lambda values: values.diff().shift(1).rolling(48, min_periods=24).std(ddof=0).to_numpy(float))
    first = front.build_age_front_state(market, dates)
    changed = market.copy()
    changed.loc[1900:, "close"] *= 10.0
    second = front.build_age_front_state(changed, dates)
    for name in ("upper_mass", "lower_mass", "upper_front", "lower_front"):
        assert np.allclose(first.loc[:1899, name], second.loc[:1899, name], equal_nan=True)


def test_front_dynamics_detects_outward_propagation():
    state = pd.DataFrame(
        {
            "upper_mass": [0.2, 0.5],
            "lower_mass": [0.0, 0.0],
            "upper_front": [0.4, 0.8],
            "lower_front": [np.nan, np.nan],
            "upper_center": [0.3, 0.6],
            "lower_center": [np.nan, np.nan],
            "upper_old_share": [0.1, 0.7],
            "lower_old_share": [np.nan, np.nan],
            "side": [1.0, 1.0],
        }
    )
    dynamics = front.build_front_dynamics(state, lag=1)
    assert dynamics.loc[1, "front_velocity"] > 0.0
    assert dynamics.loc[1, "center_velocity"] > 0.0
    assert dynamics.loc[1, "mass_growth"] > 0.0
    assert dynamics.loc[1, "propagation_score"] > 0.0
    assert np.isnan(dynamics.loc[1, "retreat_score"])


def test_fit_threshold_ignores_selection(monkeypatch):
    dates = pd.Series(pd.date_range("2022-12-31", periods=576, freq="5min"))
    monkeypatch.setitem(front.WINDOWS, "fit", ("2022-12-31", "2023-01-01"))
    values = np.linspace(0.1, 10.0, len(dates))
    first = front.fit_positive_threshold(values, dates)
    values[dates >= pd.Timestamp("2023-01-01")] = 1e9
    assert first == front.fit_positive_threshold(values, dates)


def test_policy_masks_follow_propagation_and_fade_retreat():
    dynamics = pd.DataFrame(
        {
            "side": [1, -1],
            "propagation_score": [2.0, 2.0],
            "retreat_score": [2.0, 2.0],
        }
    )
    decision = np.ones(2, dtype=bool)
    long_active, short_active = front.policy_masks(dynamics, decision, 1.0)
    retreat_long, retreat_short = front.policy_masks(dynamics, decision, 1.0, mode="retreat")
    assert long_active.tolist() == [True, False]
    assert short_active.tolist() == [False, True]
    assert np.array_equal(long_active, retreat_short)
    assert np.array_equal(short_active, retreat_long)


def test_policy_masks_require_completed_decision():
    dynamics = pd.DataFrame(
        {"side": [1, 1], "propagation_score": [2.0, 2.0], "retreat_score": [np.nan, np.nan]}
    )
    long_active, _ = front.policy_masks(dynamics, np.array([False, True]), 1.0)
    assert long_active.tolist() == [False, True]


def test_boolean_lag_has_no_wraparound():
    values = np.array([True, False, True, True])
    assert front.lag_boolean(values, 2).tolist() == [False, False, True, False]


def test_loader_keeps_2024_sealed():
    _, dates = front.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
