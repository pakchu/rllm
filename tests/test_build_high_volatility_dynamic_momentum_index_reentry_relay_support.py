import numpy as np
import pandas as pd
from training import build_high_volatility_dynamic_momentum_index_reentry_relay_support as s

def test_dynamic_period_is_bounded_and_rsi_uses_current_period():
 close=pd.Series(np.linspace(100,130,50)+np.sin(np.arange(50))*3);valid=pd.Series([True]*50);x=s.dynamic_momentum(close,valid);finite=x.dynamic_period.dropna();assert finite.between(5,30).all() and np.isfinite(x.dymi.dropna()).all()
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*7,"dymi":[20.,31.,40.,75.,69.,50.,25.],"fixed_fourteen_rsi":[20.,25.,31.,75.,69.,50.,25.],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and not a.iloc[4]
 fixed,side,_=s.active(panel(),"fixed_fourteen_rsi");assert fixed.iloc[2] and side.iloc[2]==1
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[2] and side.iloc[2]==1
