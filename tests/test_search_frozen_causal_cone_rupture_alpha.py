import numpy as np
import pandas as pd

from training import search_frozen_causal_cone_rupture_alpha as cone


def _market(n: int = 2400):
    dates = pd.Series(pd.date_range("2020-10-01", periods=n, freq="5min"))
    close = np.exp(4.0 + np.cumsum(0.001 * np.sin(np.arange(n) / 11.0)))
    return pd.DataFrame({"close": close}), dates


def test_prior_volatility_excludes_current_return(monkeypatch):
    market, _ = _market()
    monkeypatch.setattr(cone, "VOL_WINDOW", 20)
    monkeypatch.setattr(cone, "VOL_MIN_PERIODS", 10)
    log_price = np.log(market["close"])
    first = cone.prior_volatility(log_price)
    changed = log_price.copy()
    changed.iloc[100] += 2.0
    second = cone.prior_volatility(changed)
    assert first[100] == second[100]
    assert first[101] != second[101]


def test_cone_components_direction_and_mass():
    result = cone.cone_components(np.array([3.0, 4.0, -1.0, 0.0]))
    assert result["side"] == 1.0
    assert result["upper_mass"] > 0.0
    assert result["lower_mass"] == 0.0
    assert result["breach_fraction"] == 0.5


def test_cone_state_is_prefix_independent(monkeypatch):
    market, dates = _market()
    monkeypatch.setattr(cone, "VOL_WINDOW", 48)
    monkeypatch.setattr(cone, "VOL_MIN_PERIODS", 24)
    first = cone.build_cone_state(market, dates, horizon=144)
    changed = market.copy()
    changed.loc[1800:, "close"] *= 10.0
    second = cone.build_cone_state(changed, dates, horizon=144)
    assert np.allclose(first.loc[:1799, "score"], second.loc[:1799, "score"], equal_nan=True)


def test_frozen_anchor_and_current_volatility_controls_differ(monkeypatch):
    market, dates = _market()
    monkeypatch.setattr(cone, "VOL_WINDOW", 48)
    monkeypatch.setattr(cone, "VOL_MIN_PERIODS", 24)
    frozen = cone.build_cone_state(market, dates, horizon=144, frozen_anchor_volatility=True)
    current = cone.build_cone_state(market, dates, horizon=144, frozen_anchor_volatility=False)
    assert not np.allclose(frozen["score"], current["score"], equal_nan=True)


def test_fit_threshold_ignores_selection(monkeypatch):
    dates = pd.Series(pd.date_range("2022-12-31", periods=576, freq="5min"))
    monkeypatch.setitem(cone.WINDOWS, "fit", ("2022-12-31", "2023-01-01"))
    score = np.arange(len(dates), dtype=float)
    first = cone.fit_threshold(score, dates, 0.9)
    score[dates >= pd.Timestamp("2023-01-01")] = 1e9
    assert first == cone.fit_threshold(score, dates, 0.9)


def test_onset_only_suppresses_persistent_state():
    score = np.zeros(36)
    side = np.zeros(36)
    decision = np.zeros(36, dtype=bool)
    decision[[11, 23, 35]] = True
    score[[11, 23, 35]] = 2.0
    side[[11, 23, 35]] = 1.0
    long_active, _ = cone.policy_masks(score, side, decision, 1.0, onset_only=True)
    assert np.flatnonzero(long_active).tolist() == [11]


def test_loader_keeps_2024_sealed():
    _, dates = cone.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")


def test_direction_flip_is_exact():
    score = np.array([2.0, 2.0])
    side = np.array([1.0, -1.0])
    decision = np.ones(2, dtype=bool)
    long_active, short_active = cone.policy_masks(score, side, decision, 1.0)
    flip_long, flip_short = cone.policy_masks(score, side, decision, 1.0, flip=True)
    assert np.array_equal(long_active, flip_short)
    assert np.array_equal(short_active, flip_long)
