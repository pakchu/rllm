import numpy as np
import pandas as pd
from training import build_high_volatility_gator_awakening_relay_support as s

def test_smma_seed_and_recursion():
 x=s.smma(pd.Series([1.,2.,3.,4.]),pd.Series([True]*4),3);assert np.isnan(x.iloc[1]) and x.iloc[2]==2 and np.isclose(x.iloc[3],8/3)

def test_gator_visible_values_are_causally_lagged():
 old=(s.P["jaw_period"],s.P["jaw_shift"],s.P["teeth_period"],s.P["teeth_shift"],s.P["lips_period"],s.P["lips_shift"]);s.P.update(jaw_period=3,jaw_shift=2,teeth_period=2,teeth_shift=1,lips_period=2,lips_shift=0)
 try:
  h=pd.Series([10.,11.,12.,13.,14.]);l=h-2;x=s.gator(h,l,pd.Series([True]*5));assert x.visible_jaw.iloc[4]==x.raw_jaw.iloc[2] and x.visible_teeth.iloc[4]==x.raw_teeth.iloc[3] and x.visible_lips.iloc[4]==x.raw_lips.iloc[4]
 finally:s.P.update(jaw_period=old[0],jaw_shift=old[1],teeth_period=old[2],teeth_shift=old[3],lips_period=old[4],lips_shift=old[5])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"upper_gap":[1.,1.,2.,3.,2.,3.,4.],"lower_gap":[1.,1.,2.,3.,4.,3.,4.],"entry_side":[-1,-1,1,1,-1,-1,-1],"variation_rank":[.8]*7})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,True] and z[a].tolist()==[1,-1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[2]
 single,side,_=s.active(panel(),"single_gap_expansion");assert single.iloc[4] and side.iloc[4]==-1
 stale,side,_=s.active(panel(),"one_bar_stale_awakening");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()
