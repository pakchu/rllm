import numpy as np,pandas as pd
from training import build_high_volatility_flow_impact_convexity_relay_support as s
def test_rank():
 r=s.rank(pd.Series(np.arange(181,dtype=float)));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def test_metrics_convex():
 flow=np.r_[np.tile([-.1,.11],24),np.tile([-.5,.51],24)];ret=np.r_[2*flow[:48],4*flow[48:]];q=np.ones(96)*100.;buy=(flow+1)*q/2;rows=[]
 for i in range(96):
  for j in range(5):
   close=np.exp(ret[i]);rows.append({"open":1.,"high":max(1.,close),"low":min(1.,close),"close":close,"quote_asset_volume":q[i]/5,"taker_buy_quote":buy[i]/5})
 m=s.block_metrics(pd.DataFrame(rows));assert m["source_valid"];assert np.isclose(m["low_beta"],2.);assert np.isclose(m["high_beta"],4.)
def test_conditions():
 x=pd.DataFrame([{"source_valid":True,"impact_convexity":.2,"convexity_rank":.2,"variation_rank":.8,"aggregate_flow":1.,"block_return":.1},{"source_valid":True,"impact_convexity":.2,"convexity_rank":.8,"variation_rank":.8,"aggregate_flow":1.,"block_return":.1}]);a,side=s.conditions(x,"primary");assert a.tolist()==[False,True];assert side.tolist()==[1,1]
