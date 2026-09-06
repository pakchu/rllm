import numpy as np
import pandas as pd
from training import build_high_volatility_directional_variation_timing_migration_relay_support as s

def test_timing_statistics_positive_energy_arrives_later():
 returns=np.r_[np.repeat(-.001,240),np.repeat(.002,240)];open_=np.repeat(100.,480);close=open_*np.exp(returns);block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close})
 positive,negative,contrast,mass,variation=s.timing_statistics(block);assert positive>negative and contrast>0 and mass>0 and variation>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"positive_variation_centroid":[.6]*6,"negative_variation_centroid":[.4]*6,"directional_timing_contrast":[-.1,.3,.4,.2,-.5,-.4],"timing_rank":[.5,.8,.9,.4,.8,.9],"directional_variation_mass_asymmetry":[.1,-.3,-.4,-.2,.5,.4],"mass_asymmetry_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"feature_available_time":pd.date_range("2024-01-01T06:00:00Z",periods=6,freq="8h")})

def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 mass_active,mass_side,_=s.active(panel(),"directional_variation_mass_asymmetry");assert mass_active.tolist()==[False,True,False,False,True,False] and mass_side[mass_active].tolist()==[-1,1]
 forced_active,forced_side,_=s.active(panel(),"forced_long");assert forced_active.equals(active) and forced_side[forced_active].eq(1).all()

def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
