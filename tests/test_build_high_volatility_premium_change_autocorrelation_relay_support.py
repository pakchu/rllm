import hashlib
import json
import math

import numpy as np
import pandas as pd

from training import build_high_volatility_premium_change_autocorrelation_relay_support as support


def test_premium_change_autocorrelation_sign_and_invalid_denominator():
    persistent = np.cumsum(np.arange(1.0, 49.0))
    alternating = np.cumsum(np.resize(np.array([1.0, -1.0]), 48))
    assert support.premium_change_autocorrelation(persistent) > 0
    assert support.premium_change_autocorrelation(alternating) < 0
    assert math.isnan(support.premium_change_autocorrelation(np.ones(48)))
    assert math.isnan(support.premium_change_autocorrelation(np.ones(47)))


def test_prior_rank_excludes_current_and_skips_invalid(monkeypatch):
    monkeypatch.setitem(support.P, "minimum_persistence_history_decisions", 2)
    monkeypatch.setitem(support.P, "persistence_history_decisions", 3)
    rank = support.prior_rank(
        pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]),
        pd.Series([True, True, False, True, True]),
    )
    assert np.isnan(rank.iloc[1])
    assert rank.iloc[3] == 1 and rank.iloc[4] == 1


def test_onset_and_frozen_controls():
    assert support.fresh_onset(
        pd.Series([True, False, True, True]), pd.Series([True] * 4)
    ).tolist() == [False, False, True, False]
    panel = pd.DataFrame({
        "source_valid": [True] * 5,
        "variation_tail": [True, True, False, True, True],
        "persistence_tail": [False, True, True, False, True],
        "premium_change_autocorrelation": [0.1, 0.2, -0.3, -0.2, 0.4],
        "persistence_rank": [0.7, 0.9, 0.1, 0.1, 0.9],
        "direction_agreement": [True] * 5,
        "eligible": [False, True, False, False, True],
        "entry_side": [-1, 1, -1, 1, -1],
    })
    activity, side, _ = support.active(panel)
    assert activity.tolist() == [False, True, False, False, True]
    assert side[activity].tolist() == [1, -1]
    activity, _, _ = support.active(panel, "no_persistence_gate")
    assert activity.tolist() == [False, False, False, True, False]
    activity, _, _ = support.active(panel, "no_variation_gate")
    assert activity.tolist() == [False, True, False, False, True]
    activity, _, _ = support.active(panel, "negative_persistence")
    assert activity.tolist() == [False, False, False, True, False]
    activity, side, _ = support.active(panel, "one_bar_stale_onset")
    assert activity.tolist() == [False, False, True, False, False]
    assert side[activity].tolist() == [1]
    activity, side, _ = support.active(panel, "direction_flip")
    assert side[activity].tolist() == [-1, 1]
    activity, side, _ = support.active(panel, "forced_long")
    assert side[activity].eq(1).all()


def test_source_blind_and_hash_bound():
    queries = support.PREMIUM_QUERY + support.PERP_QUERY
    assert "bars_binance_premium" in support.PREMIUM_QUERY
    assert "bars_binance " in support.PERP_QUERY
    assert "funding" not in queries.lower() and "gross9" not in queries.lower()
    assert support.PREREG_SHA == hashlib.sha256(support.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    value = {"한글": "premium"}
    expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert support.canonical_hash(value) == expected
