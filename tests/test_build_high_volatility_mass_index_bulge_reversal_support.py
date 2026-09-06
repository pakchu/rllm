import numpy as np
import pandas as pd
from training import build_high_volatility_mass_index_bulge_reversal_support as s

def test_causal_ema_resets_and_warms_up():
 values=pd.Series([1.,2.,3.,4.,5.,6.]);valid=pd.Series([True,True,True,False,True,True])
 result=s.causal_ema_reset(values,valid,2)
 assert np.isnan(result.iloc[0]) and np.isfinite(result.iloc[2]) and np.isnan(result.iloc[3]) and np.isnan(result.iloc[4]) and np.isfinite(result.iloc[5])

def test_release_arms_then_fires_once():
 old_arm=s.P["bulge_arm_level"];old_release=s.P["bulge_release_level"]
 try:
  s.P["bulge_arm_level"]=27.;s.P["bulge_release_level"]=26.5
  result=s.release_state(pd.Series([26.,27.1,26.8,26.4,26.3]),pd.Series([True]*5))
  assert result.tolist()==[False,False,False,True,False]
 finally:s.P["bulge_arm_level"]=old_arm;s.P["bulge_release_level"]=old_release

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"bulge_release":[False,False,True,False,False,True],"direction":[-1,-1,1,1,-1,-1],"variation_rank":[.8,.8,.8,.8,.8,.4]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 stale,side,_=s.active(panel(),"one_bar_stale_bulge");assert stale.iloc[3] and side.iloc[3]==1
