import hashlib
import json

import numpy as np
import pandas as pd

from training import build_high_volatility_time_series_forecast_slope_flip_relay_support as s


def test_exact_lean_ols_forecast(monkeypatch):
    monkeypatch.setitem(s.P, "forecast_periods", 3)
    states = s.time_series_forecast(pd.Series([1.0, 2.0, 3.0]), pd.Series([True] * 3))
    assert np.isnan(states.forecast.iloc[1])
    assert states.forecast.iloc[2] == 4.0
    assert states.regression_slope.iloc[2] == 1.0


def test_forecast_gap_reset(monkeypatch):
    monkeypatch.setitem(s.P, "forecast_periods", 3)
    states = s.time_series_forecast(pd.Series([1., 2., 3., 4., 5., 6.]), pd.Series([True, True, True, False, True, True]))
    assert states.forecast.iloc[2] == 4.0
    assert states.forecast.iloc[3:].isna().all()


def test_prior_rank_excludes_current_and_resets(monkeypatch):
    monkeypatch.setitem(s.P, "minimum_variation_history_decisions", 2)
    monkeypatch.setitem(s.P, "variation_history_decisions", 3)
    ranks = s.prior_rank(pd.Series([1., 2., 3., np.nan, 4., 5., 6.]), pd.Series([True, True, True, False, True, True, True]))
    assert np.isnan(ranks.iloc[1]) and ranks.iloc[2] == 1
    assert np.isnan(ranks.iloc[4]) and np.isnan(ranks.iloc[5]) and ranks.iloc[6] == 1


def test_strict_flip_resets_and_controls():
    values = pd.Series([-1., 1., 2., np.nan, -1., 1.])
    valid = pd.Series([True, True, True, False, True, True])
    assert s.strict_flip_side(values, valid).tolist() == [0, 1, 0, 0, 0, 1]
    panel = pd.DataFrame({
        "source_valid": [True] * 4, "variation_rank": [.8, .8, .4, .8],
        "displacement": [-1., 2., -3., 4.], "entry_side": [-1, 1, -1, 1],
        "slope_flip_side": [1, 0, -1, 0], "flip": [True, True, True, True],
    })
    primary, side, _ = s.active(panel)
    assert primary.tolist() == [True, True, False, True] and side[primary].tolist() == [-1, 1, 1]
    assert s.active(panel, "no_variation_gate")[0].all()
    level, side, _ = s.active(panel, "forecast_level_side")
    assert level.tolist() == [True, True, False, True] and side[level].tolist() == [-1, 1, 1]
    slope, side, _ = s.active(panel, "current_regression_slope_flip")
    assert slope.tolist() == [True, False, False, False] and side[slope].tolist() == [1]
    stale, side, _ = s.active(panel, "one_bar_stale_flip")
    assert stale.tolist() == [False, True, True, False] and side[stale].tolist() == [-1, 1]
    flipped, side, _ = s.active(panel, "direction_flip")
    assert flipped.equals(primary) and side[flipped].tolist() == [1, -1, -1]
    forced, side, _ = s.active(panel, "forced_long")
    assert forced.equals(primary) and side[forced].eq(1).all()


def test_source_blind_and_hash_bound():
    query = s.QUERY.lower()
    assert "open,high,low,close" in query and "funding" not in query and "gross9" not in query
    assert s.PREREG_SHA == hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    value = {"한글": "TSF"}
    expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert s.canonical_hash(value) == expected
