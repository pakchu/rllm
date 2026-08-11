import math
import numpy as np
import pandas as pd
from training import build_high_volatility_mesa_adaptive_moving_average_crossover_relay_support as s

def test_first_mama_step_matches_lean_equations():
 state=s.LeanMama(.5,.05)
 for value in range(1,13): row=state.update(value+1,value-1)
 row=state.update(14,12)
 smooth=(4*13+3*12+2*11+10)/10
 adjusted=.54
 det=.0962*smooth*adjusted
 quad=.0962*det*adjusted
 assert row["median_price"]==13 and row["smooth"]==smooth
 assert np.isclose(row["detrender"],det) and np.isclose(row["quadrature1"],quad)
 assert np.isclose(row["period"],1.2) and row["phase"]==0 and row["alpha"]==.5
 assert np.isnan(row["mama"]) and np.isnan(row["fama"])

def test_mama_is_ready_on_frozen_33rd_bar_and_alpha_is_bounded():
 x=np.arange(40,dtype=float);high=pd.Series(100+x+np.sin(x));low=high-2
 frame=s.mama_crossover(high,low,pd.Series([True]*40))
 assert frame.mama.first_valid_index()==32 and frame.fama.first_valid_index()==32
 ready=frame.loc[32:];assert ready.alpha.between(.05,.5).all()

def test_invalid_bar_resets_all_state():
 x=np.arange(70,dtype=float);high=pd.Series(100+x);low=high-2;valid=pd.Series([True]*34+[False]+[True]*35)
 frame=s.mama_crossover(high,low,valid)
 assert frame.mama.first_valid_index()==32 and frame.loc[35:].mama.first_valid_index()==67

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"raw_median_side":[0,-1,0,1,0,-1],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 follow,side,_=s.active(panel(),"direction_flip");assert follow.iloc[2] and side.iloc[2]==-1
 raw,side,_=s.active(panel(),"raw_median_change");assert raw.iloc[1] and side.iloc[1]==-1
