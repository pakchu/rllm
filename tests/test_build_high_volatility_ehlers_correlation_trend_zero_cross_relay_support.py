import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_correlation_trend_zero_cross_relay_support as s

def test_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 values=pd.Series([1.,2.,3.,2.,1.,0.]);result=s.correlation_trend(values,pd.Series([True]*6),3);assert np.isnan(result.iloc[1]) and result.iloc[2]==1 and result.iloc[-1]==-1
 gap=s.correlation_trend(values,pd.Series([True,True,True,False,True,True]),3);assert gap.iloc[-1]!=gap.iloc[-1]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"slow_entry_side":[0,-1,0,1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 slow,side,_=s.active(panel(),"slow_40_bar_cti");assert slow.iloc[1] and side.iloc[1]==-1 and slow.iloc[3]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1

def test_primary_source_valid_does_not_depend_on_slow_control_warmup():
 base=np.repeat(np.arange(1.,41.),240);close=base*1.0001
 raw=pd.DataFrame({"ts":pd.date_range(s.START,periods=40*240,freq="1min"),"open":base,"high":close,"low":base,"close":close})
 built=s.build_panel(raw)
 assert built.loc[19,"source_valid"] and np.isnan(built.loc[19,"slow_cti"])
