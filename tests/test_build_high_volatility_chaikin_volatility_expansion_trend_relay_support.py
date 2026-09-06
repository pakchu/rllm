import hashlib
import math

import numpy as np
import pandas as pd

from training import build_high_volatility_chaikin_volatility_expansion_trend_relay_support as s


def test_range_ema_is_causal_warm_and_gap_reset(monkeypatch):
    monkeypatch.setitem(s.P, "range_ema_periods", 3)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 8.0, 10.0, 12.0])
    valid = values.notna()
    result = s.range_ema(values, valid)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 2.25
    assert result.iloc[3] == 3.125
    assert result.iloc[4:7].isna().all()
    assert result.iloc[7] == 10.5


def test_prior_rank_excludes_current(monkeypatch):
    monkeypatch.setitem(s.P, "minimum_history_decisions", 2)
    monkeypatch.setitem(s.P, "history_decisions", 3)
    rank = s.prior_rank(pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]))
    assert math.isnan(rank.iloc[1])
    assert rank.iloc[3] == 1.0
    assert rank.iloc[4] == 1.0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "chaikin_rank": [0.8, 0.8, 0.4, 0.9, 0.8, 0.9],
            "variation_rank": [0.8, 0.4, 0.8, 0.8, 0.8, 0.8],
            "raw_range_expansion": [True, True, True, False, True, True],
            "direction": [-1, 1, 1, -1, -1, 1],
            "eligible": [True, False, False, True, True, True],
        }
    )


def test_controls_are_frozen():
    active, side, _ = s.active(panel())
    assert active.tolist() == [True, False, False, True, True, True]
    assert side[active].tolist() == [-1, -1, -1, 1]
    assert s.active(panel(), "no_chaikin_gate")[0].tolist() == [True, False, True, True, True, True]
    assert s.active(panel(), "no_variation_gate")[0].tolist() == [True, True, False, True, True, True]
    assert s.active(panel(), "raw_range_expansion")[0].tolist() == [True, False, True, False, True, True]
    stale, stale_side, _ = s.active(panel(), "one_bar_stale_state")
    assert stale.tolist() == [False, True, False, False, True, True]
    assert stale_side[stale].tolist() == [-1, -1, -1]
    flipped, flipped_side, _ = s.active(panel(), "direction_flip")
    assert flipped_side[flipped].tolist() == [1, 1, 1, -1]
    forced, forced_side, _ = s.active(panel(), "forced_long")
    assert forced_side[forced].eq(1).all()


def test_source_is_blind_and_prereg_hash_bound():
    assert "bars_binance " in s.QUERY
    assert "funding" not in s.QUERY.lower()
    assert "gross9" not in s.QUERY.lower()
    assert s.PREREG_SHA == hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
