import numpy as np
import pandas as pd

from training import build_high_volatility_directional_movement_index_crossover_relay_support as support


def test_prior_rank_excludes_current() -> None:
    ranks = support.prior_rank(pd.Series(np.arange(121, dtype=float)))
    assert np.isnan(ranks.iloc[119])
    assert ranks.iloc[120] == 1


def test_wilder_directional_movement_and_adx_seed() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="4h", tz="UTC")
    high = pd.Series(np.arange(100, 130, dtype=float), index=index)
    low = high - 2
    close = high - 1
    valid = pd.Series(True, index=index)
    result = support.wilder_adx(high, low, close, valid)
    assert result.plus_dm.iloc[1] == 1
    assert result.minus_dm.iloc[1] == 0
    assert result.plus_di.iloc[14] > 0
    assert result.minus_di.iloc[14] == 0
    assert result.dx.iloc[14] == 100
    assert result.adx.iloc[27] == 100


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"source_valid": [True] * 5, "plus_di": [30, 10, 20, 40, 10],
         "minus_di": [10, 30, 20, 10, 40], "difference": [-20, 20, 10, -30, 30],
         "variation_rank": [.8, .8, .8, .8, .4]}
    )


def test_primary_and_controls() -> None:
    active,side,_=support.active(frame());assert active.tolist()==[False,True,False,True,False] and side[active].tolist()==[1,-1]
    assert support.active(frame(),"no_variation_gate")[0].iloc[4]
    assert support.active(frame(),"persistent_direction_state")[0].iloc[2]
    stale,stale_side,_=support.active(frame(),"one_bar_stale_cross");assert stale.iloc[2] and stale_side.iloc[2]==1
    flipped=support.active(frame(),"direction_flip")[1];assert flipped[active].tolist()==[-1,1]
