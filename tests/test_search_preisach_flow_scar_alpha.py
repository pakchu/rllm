import numpy as np
import pandas as pd

from training import search_preisach_flow_scar_alpha as preisach


def _market(n: int = 300):
    close = np.exp(4.0 + np.cumsum(0.002 * np.sin(np.arange(n) / 9.0)))
    quote = 1000.0 + np.arange(n)
    buy = 0.55 * quote
    return pd.DataFrame(
        {
            "close": close,
            "quote_asset_volume": quote,
            "taker_buy_quote": buy,
        }
    )


def test_relay_lattice_is_fixed_and_valid():
    alpha, beta = preisach.relay_lattice()
    assert len(alpha) == 12
    assert np.all(alpha > beta)
    assert set(np.round(alpha - beta, 10)) == {1.0, 1.5, 2.0}
    assert preisach.MIN_AVALANCHE == 3


def test_input_center_scale_and_flow_denominator_are_prior_only(monkeypatch):
    market = _market()
    monkeypatch.setattr(preisach, "FIELD_WINDOW", 24)
    monkeypatch.setattr(preisach, "FIELD_MIN_PERIODS", 12)
    monkeypatch.setattr(preisach, "FLOW_DENOM_WINDOW", 6)
    first = preisach.build_inputs(market)
    changed = market.copy()
    changed.loc[100, "close"] *= 2.0
    changed.loc[100, "quote_asset_volume"] *= 3.0
    second = preisach.build_inputs(changed)
    assert first.loc[100, "prior_price_center"] == second.loc[100, "prior_price_center"]
    assert first.loc[100, "prior_price_scale"] == second.loc[100, "prior_price_scale"]
    assert first.loc[100, "prior_hour_quote"] == second.loc[100, "prior_hour_quote"]
    assert first.loc[100, "price_field"] != second.loc[100, "price_field"]
    assert first.loc[100, "signed_flow"] != second.loc[100, "signed_flow"]


def test_relay_retains_state_inside_deadband():
    up_path = np.array([3.0, 0.25])
    down_path = np.array([-3.0, 0.25])
    flow = np.zeros(2)
    up = preisach.run_relay_ensemble(up_path, flow)
    down = preisach.run_relay_ensemble(down_path, flow)
    assert up.loc[1, "magnetization"] != down.loc[1, "magnetization"]


def test_release_pressure_excludes_current_bar_flow():
    # Initialize down, accumulate bullish flow, then flip up. The opposing scar
    # is prior negative pressure only, so changing current flow cannot rewrite it.
    path = np.array([-3.0, -3.0, 3.0])
    prior_negative = np.array([0.0, -1.0, 0.0])
    current_positive = np.array([0.0, -1.0, 100.0])
    first = preisach.run_relay_ensemble(path, prior_negative)
    second = preisach.run_relay_ensemble(path, current_positive)
    assert first.loc[2, "opposing_pressure"] == second.loc[2, "opposing_pressure"]
    assert first.loc[2, "release_score"] > second.loc[2, "release_score"]


def test_relay_engine_is_prefix_independent():
    x = np.sin(np.arange(200) / 7.0) * 3.0
    flow = np.cos(np.arange(200) / 5.0) * 0.1
    first = preisach.run_relay_ensemble(x, flow)
    changed_x = x.copy()
    changed_flow = flow.copy()
    changed_x[150:] *= -10.0
    changed_flow[150:] *= 100.0
    second = preisach.run_relay_ensemble(changed_x, changed_flow)
    for column in first.columns:
        assert np.allclose(first.loc[:149, column], second.loc[:149, column], equal_nan=True)


def test_fit_threshold_ignores_selection(monkeypatch):
    dates = pd.Series(pd.date_range("2022-12-31", periods=576, freq="5min"))
    monkeypatch.setitem(preisach.WINDOWS, "fit", ("2022-12-31", "2023-01-01"))
    state = pd.DataFrame(
        {
            "direction": np.ones(len(dates)),
            "avalanche_count": np.full(len(dates), 3.0),
            "coherence": np.ones(len(dates)),
            "release_score": np.linspace(0.1, 10.0, len(dates)),
        }
    )
    first = preisach.fit_score_threshold(state, dates, "release_score")
    state.loc[dates >= pd.Timestamp("2023-01-01"), "release_score"] = 1e9
    assert first == preisach.fit_score_threshold(state, dates, "release_score")


def test_policy_direction_flip_is_exact():
    state = pd.DataFrame(
        {
            "direction": [1.0, -1.0],
            "avalanche_count": [3.0, 3.0],
            "coherence": [1.0, 1.0],
            "release_score": [2.0, 2.0],
        }
    )
    long_active, short_active = preisach.policy_masks(state, 1.0)
    flip_long, flip_short = preisach.policy_masks(state, 1.0, flip=True)
    assert np.array_equal(long_active, flip_short)
    assert np.array_equal(short_active, flip_long)


def test_lag_values_has_no_wraparound():
    values = np.array([True, False, True, True])
    assert preisach.lag_values(values, 2, fill=False).tolist() == [False, False, True, False]


def test_loader_keeps_2024_sealed():
    _, dates = preisach.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
