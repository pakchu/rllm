import numpy as np
import pandas as pd
from training import build_high_volatility_macd_crossover_relay_support as s


def test_seeded_ema_uses_arithmetic_seed_then_canonical_recursion():
    values=pd.Series([1.,2.,3.,4.]);out=s.seeded_ema(values,3)
    assert np.isnan(out.iloc[1]) and out.iloc[2]==2
    assert out.iloc[3]==3


def test_prior_rank_excludes_current():
    ranks=s.prior_rank(pd.Series(np.arange(121,dtype=float)))
    assert np.isnan(ranks.iloc[119]) and ranks.iloc[120]==1


def panel():
    return pd.DataFrame({"source_valid":[True]*6,"difference":[-2.,-1.,1.,2.,-1.,-2.],"variation_rank":[.8,.8,.8,.8,.4,.8]})


def test_primary_and_controls():
    active,side,_=s.active(panel())
    assert active.tolist()==[False,False,True,False,False,False]
    assert side[active].tolist()==[1]
    no_vol=s.active(panel(),"no_variation_gate")[0]
    assert no_vol.iloc[4]
    stale,stale_side,_=s.active(panel(),"one_day_stale_crossover")
    assert stale.iloc[3] and stale_side.iloc[3]==1
    flipped=s.active(panel(),"direction_flip")[1]
    assert flipped[active].tolist()==[-1]
