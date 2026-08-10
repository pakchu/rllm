import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_center_of_gravity_crossover_relay_support as s

def test_center_of_gravity_and_cross_side():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 old=s.P["cog_periods"];s.P["cog_periods"]=3
 try:
  flat=s.center_of_gravity(pd.Series([10.,10.,10.,10.]),pd.Series([True]*4));assert np.isnan(flat.iloc[1]) and flat.iloc[2]==0
  rising=s.center_of_gravity(pd.Series([10.,11.,12.]),pd.Series([True]*3));assert rising.iloc[2]>0
 finally:s.P["cog_periods"]=old

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"zero_line_side":[0,0,1,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
