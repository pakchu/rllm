import numpy as np
import pandas as pd
from training import build_high_volatility_robust_daily_return_outlier_reversal_support as s
def test_robust_score_uses_prior_median_and_mad():
 prior=np.arange(30,dtype=float);median,mad,score=s.robust_score(40.,prior)
 assert median==14.5 and mad==7.5 and score>0
def panel():
 return pd.DataFrame({"source_valid":[True]*6,"robust_score":[-1.,1.,1.,-1.,-1.,1.],"outlier_strength_rank":[.5,.8,.8,.5,.8,.8],"current_return":[1.,1.,-1.,-1.,1.,-1.],"raw_return_strength_rank":[.8]*6,"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01 02:00",periods=6,freq="1d",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[-1,-1,1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 raw_active,raw_side,_=s.active(panel(),"raw_return_fade");assert raw_active.all() and raw_side.tolist()==[-1,-1,1,1,-1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(ranks.iloc[119]) and ranks.iloc[120]==1.
