import numpy as np
import pandas as pd
from training import build_high_volatility_flat_auction_absorption_relay_support as s

def test_flat_statistics():
 open_=np.repeat(100.,480);close=open_.copy();close[10:]=100*np.exp(np.linspace(.0001,.03,470));quote=np.ones(480);quote[:10]=10
 block=pd.DataFrame({"open":open_,"high":np.maximum(open_,close),"low":np.minimum(open_,close),"close":close,"quote_asset_volume":quote})
 share,count_share,var,ret,final_ret,count=s.flat_statistics(block)
 assert count==10 and share>count_share and var>0 and ret>0 and final_ret>0

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"flat_turnover_share":[.1,.3,.4,.2,.5,.4],"flat_share_rank":[.5,.8,.9,.4,.8,.9],"flat_minute_count_share":[.1,.3,.4,.2,.5,.4],"flat_count_share_rank":[.5,.8,.9,.4,.8,.9],"variation_rank":[.8]*6,"completed_return":[.01,.02,.03,-.01,-.02,.01],"final_two_hour_return":[.01,.01,.02,-.01,-.01,.01],"feature_available_time":pd.date_range("2024-01-01",periods=6,freq="8h",tz="UTC")})
def test_primary_and_controls():
 active,side,_=s.active(panel());assert active.tolist()==[False,True,False,False,True,False] and side[active].tolist()==[1,-1]
 x=panel();x.loc[1,"variation_rank"]=.4;assert s.active(x,"no_variation_gate")[0].iloc[1]
 x=panel();x.loc[1,"flat_share_rank"]=.4;assert s.active(x,"flat_minute_count_share")[0].iloc[1]
def test_prior_rank_excludes_current():
 ranks=s.prior_rank(pd.Series(np.arange(181,dtype=float)));assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.
