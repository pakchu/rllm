import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_ultimate_oscillator_rate_turn_relay_support as s

def test_high_pass_and_ultimate_warmup_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 prices=pd.Series(100+np.sin(np.arange(230)/4)+np.arange(230)*.01);valid=pd.Series([True]*110+[False]+[True]*119);hp=s.high_pass(prices,valid,20);x=s.ultimate_oscillator(prices,valid,20)
 assert hp.high_pass.iloc[:3].tolist()==[0.,0.,0.] and np.isfinite(hp.high_pass.iloc[3]) and np.isnan(hp.high_pass.iloc[110])
 assert x.run_length.iloc[111:114].tolist()==[1,2,3] and x.ultimate_oscillator.iloc[209]!=x.ultimate_oscillator.iloc[209] and np.isfinite(x.ultimate_oscillator.iloc[210]) and np.isfinite(x.rate.iloc[211])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"chart_entry_side":[0,-1,0,1,0,0],"oscillator_entry_side":[0,0,-1,0,1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"chart_30_band_edge");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 osc,side,_=s.active(panel(),"oscillator_zero_cross");assert osc.iloc[2] and side.iloc[2]==-1
