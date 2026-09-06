import numpy as np
import pandas as pd
from training import build_high_volatility_dual_half_signed_volume_persistence_relay_support as s

def test_persistence_statistics_positive_both_halves():
 open_=np.repeat(100.,480);returns=np.r_[np.repeat(.001,180),np.repeat(-.001,60),np.repeat(.001,160),np.repeat(-.001,80)];close=open_*np.exp(returns);quote=np.where(returns>0,2.,1.)
 block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close,"quote_asset_volume":quote})
 first,second,weak,full,first_breadth,second_breadth,equal_weak,variation=s.persistence_statistics(block)
 assert first>0 and second>0 and weak>0 and full>0 and first_breadth>0 and second_breadth>0 and equal_weak>0 and variation>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"first_half_balance":[-.1,.3,.4,.2,-.5,-.4],"second_half_balance":[-.2,.2,.3,-.2,-.4,-.3],"weak_half_strength":[.1,.2,.3,.2,.4,.3],"strength_rank":[.5,.8,.9,.4,.8,.9],"full_block_balance":[-.1,.3,.4,.2,-.5,-.4],"full_balance_rank":[.5,.8,.9,.4,.8,.9],"first_half_sign_breadth":[.1,-.3,-.4,-.2,.5,.4],"second_half_sign_breadth":[.2,-.2,-.3,.2,.4,.3],"equal_weight_strength":[.1,.2,.3,.2,.4,.3],"equal_weight_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})

def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,True,False,True,True] and side[active].tolist()==[1,1,-1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 equal_active,equal_side,_=s.active(panel(),"equal_weight_half_sign_breadth");assert equal_active.tolist()==[False,True,True,False,True,True] and equal_side[equal_active].tolist()==[-1,-1,1,1]
 forced_active,forced_side,_=s.active(panel(),"same_clock_forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
