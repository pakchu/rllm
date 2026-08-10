import numpy as np
import pandas as pd
from training import build_high_volatility_wilder_rsi_reversal_support as s

def test_wilder_seed_and_recursion():
 changes=pd.Series([np.nan]+[1.0]*14+[-1.0]);g,l,r=s.wilder_rsi(changes)
 assert g.iloc[14]==1 and l.iloc[14]==0 and r.iloc[14]==100
 assert g.iloc[15]==13/14 and l.iloc[15]==1/14 and 90<r.iloc[15]<100

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(ranks.iloc[119]) and ranks.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*5,"rsi":[20.,80.,50.,25.,75.],"variation_rank":[.8,.8,.8,.4,.8],"feature_available_time":pd.date_range('2024-01-01',periods=5,tz='UTC')})

def test_primary_controls():
 active,side,_=s.active(panel());assert active.tolist()==[True,True,False,False,True] and side[active].tolist()==[1,-1,-1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[3]
 a,z,_=s.active(panel(),"forced_long");assert a.equals(active) and z[a].eq(1).all()
