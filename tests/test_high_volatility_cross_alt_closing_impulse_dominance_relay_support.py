import pandas as pd
import numpy as np
from training import build_high_volatility_cross_alt_closing_impulse_dominance_relay_support as s


def test_four_clean_positive_impulses_define_long_side():
    up = pd.DataFrame([[1, 1, 1, 1, 0, 0]], dtype=bool)
    down = pd.DataFrame([[0, 0, 0, 0, 0, 0]], dtype=bool)
    side, pos, neg = s.impulse_side(up, down, 4)
    assert side.iloc[0] == 1 and pos.iloc[0] == 4 and neg.iloc[0] == 0


def test_opposite_impulse_invalidates_direction():
    up = pd.DataFrame([[1, 1, 1, 1, 0, 0]], dtype=bool)
    down = pd.DataFrame([[0, 0, 0, 0, 1, 0]], dtype=bool)
    assert s.impulse_side(up, down, 4)[0].iloc[0] == 0


def test_primary_follows_common_impulse():
    panel = pd.DataFrame({"source_valid":[True,True],"impulse_side":[0,-1],"positive_impulse_count":[0,0],"negative_impulse_count":[0,4],"whole_bar_side":[0,-1],"variation_active":[True,True]})
    active, side, _ = s.active(panel)
    assert active.tolist() == [False, True] and side.iloc[1] == -1


def test_pinned_registration():
    assert s.PREREG_SHA == "e42e4174413c2062968c63ba340811e9623ada9951cae8c756c316a40b1192ab"


def test_fast_midrank_matches_strict_prior_definition():
    values = pd.Series([1.0, 2.0, 2.0, np.nan, 3.0, 1.0, 4.0])
    actual = s.strict_prior_midrank(values, window=3, minimum=2)
    expected = pd.Series([np.nan, np.nan, 0.75, np.nan, 1.0, 0.0, 1.0])
    pd.testing.assert_series_equal(actual, expected)
