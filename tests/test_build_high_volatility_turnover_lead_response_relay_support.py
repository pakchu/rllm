import numpy as np
import pandas as pd
from training import build_high_volatility_turnover_lead_response_relay_support as s

def test_turnover_statistics_pairs_current_turnover_with_next_return():
 q=np.linspace(1,96,96);lead=np.log1p(q[:-1]);ret=np.r_[0.,(lead-lead.mean())/lead.std()*.01]
 block=pd.DataFrame({"open":np.repeat(100.,480),"close":np.repeat(100*np.exp(ret),5),"quote_asset_volume":np.repeat(q/5,5)})
 lead,contemp,var,completed=s.turnover_statistics(block)
 assert lead>.99 and np.isfinite(contemp) and var>0 and np.isfinite(completed)

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"lead_response":[.1,.3,.4,-.2,-.5,.4],"response_strength_rank":[.5,.8,.9,.4,.8,.9],"contemporaneous_response":[.2,.1,.2,-.3,-.4,.2],"contemporaneous_strength_rank":[.5,.4,.6,.8,.9,.6],"variation_rank":[.8]*6,"completed_return":[.01,.02,.03,-.01,-.02,.01],"feature_available_time":pd.date_range('2024-01-01',periods=6,freq='8h',tz='UTC')})

def test_primary_onset_and_side():
 active,side,_=s.active(panel(),'primary');assert active.tolist()==[False,True,False,False,True,False];assert side[active].tolist()==[1,-1]
def test_controls_are_diagnostic():
 x=panel();x.loc[1,'variation_rank']=.4;assert s.active(x,'no_variation_gate')[0].iloc[1]
 active,side,_=s.active(panel(),'direction_flip');assert side[active].tolist()==[-1,1]
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(r.iloc[179]) and r.iloc[180]==1.
