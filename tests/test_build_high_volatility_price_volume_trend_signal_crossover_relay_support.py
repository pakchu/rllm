import hashlib

import numpy as np
import pandas as pd

from training import build_high_volatility_price_volume_trend_signal_crossover_relay_support as s


def test_prior_rank_excludes_current(monkeypatch):
    monkeypatch.setitem(s.P, "minimum_variation_history_decisions", 2)
    monkeypatch.setitem(s.P, "variation_history_decisions", 3)
    rank = s.prior_rank(pd.Series([1.0, 2.0, np.nan, 3.0]))
    assert np.isnan(rank.iloc[1]) and rank.iloc[3] == 1.0


def test_causal_pvt_uses_full_return_magnitude_and_resets():
    result = s.causal_pvt(
        pd.Series([10.0, 11.0, 13.2, 9.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([True, True, True, False]),
    )
    assert np.isclose(result.volume_return_contribution.iloc[1], 0.2)
    assert np.isclose(result.volume_return_contribution.iloc[2], 0.6)
    assert np.isclose(result.pvt.iloc[2], 0.8)
    assert np.isnan(result.pvt.iloc[3])


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "difference": [-2.0, -1.0, 1.0, 2.0, -1.0, -2.0],
            "contribution_side": [0, 0, 1, 0, -1, 0],
            "variation_rank": [0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
        }
    )


def test_controls():
    active, side, _ = s.active(panel())
    assert active.tolist() == [False, False, True, False, False, False]
    assert side[active].tolist() == [1]
    assert s.active(panel(), "no_variation_gate")[0].iloc[4]
    stale, stale_side, _ = s.active(panel(), "one_bar_stale_cross")
    assert stale.iloc[3] and stale_side.iloc[3] == 1
    raw, raw_side, _ = s.active(panel(), "contribution_zero_cross")
    assert raw.iloc[2] and raw_side.iloc[2] == 1


def test_source_is_blind_and_hash_bound():
    assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
    assert s.PREREG_SHA == hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
