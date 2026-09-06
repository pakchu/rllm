import numpy as np
import pandas as pd
from training import build_high_volatility_momersion_regime_switch_relay_support as s

def test_momersion_counts_positive_and_negative_change_products():
 close=pd.Series([0.,1.,2.,1.,2.,3.,2.,1.,2.,3.,2.,3.]);f=s.momersion_regime_switch(close,pd.Series([True]*len(close)))
 products=np.diff(close.to_numpy())[:-1]*np.diff(close.to_numpy())[1:]
 assert f.momersion_ready.iloc[11] and np.isclose(f.momersion.iloc[11],100*np.sum(products>0)/10)

def test_first_ready_cross_above_follows_current_change():
 close=pd.Series(np.arange(12,dtype=float));f=s.momersion_regime_switch(close,pd.Series([True]*12))
 assert f.momersion.iloc[10]==50 and f.momersion.iloc[11]==100 and f.entry_side.iloc[11]==1

def test_zero_density_falls_back_to_fifty():
 close=pd.Series([1.,1.,2.,2.,3.,3.,4.,4.,5.,5.,6.,6.]);f=s.momersion_regime_switch(close,pd.Series([True]*12))
 assert f.momersion_ready.iloc[11] and f.momersion.iloc[11]==50

def test_invalid_bar_resets_product_readiness():
 close=pd.Series(np.arange(30,dtype=float));valid=pd.Series([True]*13+[False]+[True]*16);f=s.momersion_regime_switch(close,valid)
 assert f.loc[14:].momersion_ready.idxmax()==25

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_one_bar_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_switch");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 raw,side,_=s.active(panel(),"raw_one_bar_direction");assert raw.iloc[1] and side.iloc[1]==-1
