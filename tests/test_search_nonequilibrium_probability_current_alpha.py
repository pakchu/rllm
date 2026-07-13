import numpy as np
import pandas as pd

from training import search_nonequilibrium_probability_current_alpha as current


def _market(n: int = 2400):
    dates = pd.Series(pd.date_range("2020-10-01", periods=n, freq="5min"))
    price = np.exp(4.0 + np.cumsum(0.001 * np.sin(np.arange(n) / 13.0)))
    quote = 1000.0 + 50.0 * np.cos(np.arange(n) / 7.0)
    buy = quote * (0.5 + 0.1 * np.sin(np.arange(n) / 11.0))
    oi = np.exp(10.0 + np.cumsum(0.0002 * np.cos(np.arange(n) / 17.0)))
    return pd.DataFrame(
        {
            "close": price,
            "quote_asset_volume": quote,
            "taker_buy_quote": buy,
            "open_interest": oi,
        }
    ), dates


def test_microstates_use_completed_hour_and_encode_eight_states():
    market, dates = _market()
    hourly = current.build_hourly_microstates(market, dates)
    assert hourly.attrs["state_count"] == 8
    assert set(hourly.loc[hourly.state >= 0, "state"].unique()).issubset(set(range(8)))
    assert np.all(hourly["position"].to_numpy() % 12 == 11)


def test_transition_counts_exclude_current_transition():
    states = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int16)
    first = current.transition_counts_before(states, 5, state_count=3, window=10)
    changed = states.copy()
    changed[5] = 0
    second = current.transition_counts_before(changed, 5, state_count=3, window=10)
    assert np.array_equal(first, second)
    assert first.sum() == 4


def test_probability_current_detects_directed_cycle():
    joint = np.full((3, 3), 0.001)
    joint[0, 1] = 0.30
    joint[1, 2] = 0.30
    joint[2, 0] = 0.30
    joint /= joint.sum()
    velocity, strength, entropy = current._current_projection(
        joint, 0, np.array([-1.0, 1.0, -1.0])
    )
    assert velocity > 0.9
    assert strength > 0.0
    assert entropy > 0.0


def test_transition_features_are_prefix_independent(monkeypatch):
    market, dates = _market(4000)
    hourly = current.build_hourly_microstates(market, dates)
    monkeypatch.setattr(current, "MIN_TRANSITIONS", 20)
    first = current.build_transition_features(hourly, window=40)
    changed = hourly.copy()
    changed.loc[250:, "state"] = (changed.loc[250:, "state"] + 3) % 8
    changed.attrs = hourly.attrs.copy()
    second = current.build_transition_features(changed, window=40)
    for column in ("current_score", "current_direction", "markov_score"):
        assert np.allclose(first.loc[:249, column], second.loc[:249, column], equal_nan=True)


def test_fit_threshold_ignores_selection(monkeypatch):
    features = pd.DataFrame(
        {
            "date": pd.date_range("2022-12-31", periods=576, freq="5min"),
            "current_score": np.linspace(0.1, 10.0, 576),
        }
    )
    monkeypatch.setitem(current.WINDOWS, "fit", ("2022-12-31", "2023-01-01"))
    first = current.fit_threshold(features, "current_score")
    features.loc[features.date >= pd.Timestamp("2023-01-01"), "current_score"] = 1e9
    assert first == current.fit_threshold(features, "current_score")


def test_policy_masks_map_only_hourly_positions():
    features = pd.DataFrame(
        {
            "position": [11, 23],
            "current_score": [2.0, 2.0],
            "current_velocity": [1.0, -1.0],
            "current_direction": [1.0, -1.0],
            "price_side": [-1, 1],
        }
    )
    long_active, short_active = current.policy_masks(features, 30, 1.0)
    assert np.flatnonzero(long_active).tolist() == [11]
    assert np.flatnonzero(short_active).tolist() == [23]


def test_boolean_lag_has_no_wraparound():
    values = np.array([True, False, True, True])
    assert current.lag_boolean(values, 2).tolist() == [False, False, True, False]


def test_loader_keeps_2024_sealed():
    _, dates = current.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
