import numpy as np
import pandas as pd
from training import build_high_volatility_range_expansion_index_reentry_relay_support as s

def test_td_rei_filters_and_bounds():
 h=pd.Series([20.,20.,20.,20.,20.,20.,10.,20.,15.,16.,17.,18.,19.,20.,21.,22.])
 l=pd.Series([5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.])
 c=pd.Series([20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.])
 x=s.td_rei(h,l,c,pd.Series([True]*len(h)))
 assert x.high_filter_zero.iloc[8]
 finite=x.rei.dropna();assert finite.between(-100,100).all()

def test_td_rei_unfiltered_identity_when_all_moves_admitted():
 h=pd.Series(np.arange(20,dtype=float)+200);l=pd.Series(100-np.arange(20,dtype=float));c=pd.Series([150.]*20);v=pd.Series([True]*20)
 x=s.td_rei(h,l,c,v);assert np.allclose(x.rei.dropna(),x.unfiltered_rei.dropna())

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"rei":[-70.,-65.,-50.,0.,70.,50.,0.],"unfiltered_rei":[-70.,-65.,-50.,0.,70.,50.,0.],"variation_rank":[.8,.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[5]
 unfiltered,side,_=s.active(panel(),"unfiltered_rei");assert unfiltered.iloc[2] and side.iloc[2]==1
 stale,side,_=s.active(panel(),"one_bar_stale_reentry");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()
