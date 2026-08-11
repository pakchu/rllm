import numpy as np
import pandas as pd
from training import build_high_volatility_pivot_points_high_low_reversal_relay_support as s

def frame(high,low=None,valid=None):
 high=pd.Series(high,dtype=float);low=pd.Series(low if low is not None else np.asarray(high)-1,dtype=float);open_=pd.Series((high+low)/2);close=open_+.1
 return s.pivot_points_high_low(open_,high,low,close,pd.Series(valid if valid is not None else [True]*len(high)))

def test_strict_high_is_confirmed_ten_bars_later_and_faded():
 high=list(range(10,20))+[30]+list(range(20,30));f=frame(high)
 assert f.pivot_type.first_valid_index()==20 and f.loc[20,"pivot_type"]==1 and f.loc[20,"entry_side"]==-1 and f.loc[20,"pivot_center_high"]==30

def test_strict_low_is_confirmed_and_tie_is_rejected():
 low=list(range(20,10,-1))+[0]+list(range(1,11));high=np.asarray(low)+2;f=frame(high,low)
 assert f.loc[20,"pivot_type"]==-1 and f.loc[20,"entry_side"]==1
 low[0]=low[10];assert frame(high,low).loc[20,"pivot_type"]==0

def test_invalid_bar_resets_twenty_one_bar_window():
 high=list(range(80));valid=[True]*22+[False]+[True]*57;f=frame(high,valid=valid)
 assert f.pivot_type.first_valid_index()==20 and f.loc[23:].pivot_type.first_valid_index()==43

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_bar_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_confirmation");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 raw,side,_=s.active(panel(),"raw_bar_direction");assert raw.iloc[1] and side.iloc[1]==-1
