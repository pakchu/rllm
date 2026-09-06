import numpy as np
import pandas as pd
from training import build_high_volatility_wilder_swing_index_zero_cross_relay_support as s

def test_wilder_swing_index_matches_lean_branch_formula():
 open_=pd.Series([10.,11.]);high=pd.Series([12.,15.]);low=pd.Series([9.,12.]);close=pd.Series([11.,14.])
 frame=s.wilder_swing_index(open_,high,low,close,pd.Series([True,True]))
 assert frame.loc[1,"n_value"]==4.75
 assert frame.loc[1,"r_value"]==3.75
 assert frame.loc[1,"k_value"]==4.
 assert np.isclose(frame.loc[1,"swing_index"],50*(4.75/3.75)*(4/20))

def test_positive_limit_move_only_rescales_swing_index():
 open_=pd.Series([10.,11.,14.]);high=pd.Series([12.,15.,15.]);low=pd.Series([9.,12.,9.]);close=pd.Series([11.,14.,10.]);valid=pd.Series([True]*3)
 at_20=s.wilder_swing_index(open_,high,low,close,valid,20).swing_index
 at_40=s.wilder_swing_index(open_,high,low,close,valid,40).swing_index
 assert np.allclose(at_20.dropna(),2*at_40.dropna())
 assert np.array_equal(np.sign(at_20.dropna()),np.sign(at_40.dropna()))

def test_wilder_swing_index_resets_after_invalid_bar():
 open_=pd.Series([10.,11.,12.,13.]);high=open_+2;low=open_-1;close=open_+.5
 frame=s.wilder_swing_index(open_,high,low,close,pd.Series([True,True,False,True]))
 assert np.isfinite(frame.loc[1,"swing_index"])
 assert np.isnan(frame.loc[2,"swing_index"]) and np.isnan(frame.loc[3,"swing_index"])

def test_raw_close_change_control_tracks_sign_not_prior_event():
 open_=pd.Series([10.,11.,12.,13.]);high=open_+2;low=open_-1;close=open_+.5
 frame=s.wilder_swing_index(open_,high,low,close,pd.Series([True]*4))
 assert frame.raw_close_side.tolist()==[0,1,0,0]

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_close_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 zero,side,_=s.active(panel(),"raw_close_change");assert zero.iloc[1] and side.iloc[1]==-1
