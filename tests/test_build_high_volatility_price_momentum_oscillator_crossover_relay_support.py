import numpy as np
import pandas as pd
from training import build_high_volatility_price_momentum_oscillator_crossover_relay_support as s

def test_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 old=dict(s.P);s.P.update(first_smoothing_periods=2,second_smoothing_periods=2,signal_periods=2,first_scale=10.)
 try: values=s.pmo_values(pd.Series([1.,2.,3.,4.,5.,6.,7.]),pd.Series([True]*7));assert np.isnan(values.pmo.iloc[2]) and values.pmo.iloc[-1]>0 and np.isfinite(values.difference.iloc[-1])
 finally: s.P.clear();s.P.update(old)

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"zero_line_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
