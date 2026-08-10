import numpy as np
import pandas as pd
from training import build_high_volatility_directional_turnover_concentration_asymmetry_relay_support as s

def test_concentration_statistics_detects_up_side_concentration():
 signs=np.where(np.arange(96)%2==0,1.,-1.);group_turnover=np.where(signs>0,1.,10.);group_turnover[0]=1000.
 open_=np.repeat(100.,480);close=open_.copy();turnover=np.empty(480)
 for group,sign in enumerate(signs):
  close[group*5:(group+1)*5]=100.*np.exp(sign*.001);turnover[group*5:(group+1)*5]=group_turnover[group]/5
 block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close,"quote_asset_volume":turnover})
 up,down,contrast,mass,variation=s.concentration_statistics(block);assert up>down and contrast>0 and mass>0 and variation>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"up_concentration":[.6]*6,"down_concentration":[.4]*6,"directional_concentration_contrast":[-.1,.3,.4,.2,-.5,-.4],"concentration_rank":[.5,.8,.9,.4,.8,.9],"directional_turnover_mass_asymmetry":[.1,-.3,-.4,-.2,.5,.4],"mass_asymmetry_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T02:00:00Z",periods=6,freq="8h")})

def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 mass_active,mass_side,_=s.active(panel(),"directional_turnover_mass_asymmetry");assert mass_active.tolist()==[False,True,False,False,True,False] and mass_side[mass_active].tolist()==[-1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
