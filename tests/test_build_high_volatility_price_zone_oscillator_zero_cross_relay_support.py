import numpy as np
import pandas as pd
from training import build_high_volatility_price_zone_oscillator_zero_cross_relay_support as support


def test_seeded_ema_uses_sma_seed_and_resets_on_gap():
    values = pd.Series([1.0] * 14 + [2.0, np.nan] + [3.0] * 14)
    valid = values.notna()
    result = support.seeded_ema(values, valid, 14)
    assert result.iloc[:13].isna().all()
    assert result.iloc[13] == 1.0
    assert result.iloc[14] == 1.0 + 2.0 / 15.0
    assert result.iloc[15:29].isna().all()
    assert result.iloc[29] == 3.0


def test_prior_rank_excludes_current():
    original = support.P
    support.P = {**original, "variation_history_decisions": 5, "minimum_variation_history_decisions": 3}
    try:
        ranks = support.prior_rank(pd.Series([1.0, 2.0, 3.0, 4.0]))
    finally:
        support.P = original
    assert ranks.iloc[:3].isna().all()
    assert ranks.iloc[3] == 1.0


def test_active_primary_and_controls():
    panel = pd.DataFrame({"source_valid": [True] * 4, "cross_side": [0, 1, 0, -1], "pzo_cross": [False, True, False, True], "signed_close": [-1.0, 1.0, 1.0, -1.0], "variation_rank": [0.8, 0.8, 0.8, 0.4]})
    selected, side = support.active(panel)
    assert selected.tolist() == [False, True, False, False]
    assert side[selected].tolist() == [1]
    no_gate, _ = support.active(panel, "no_variation_gate")
    assert no_gate.tolist() == [False, True, False, True]
    flipped, flipped_side = support.active(panel, "direction_flip")
    assert flipped.equals(selected)
    assert flipped_side[flipped].tolist() == [-1]
