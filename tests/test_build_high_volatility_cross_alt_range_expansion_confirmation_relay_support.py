import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_range_expansion_confirmation_relay_support as s
def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(61,dtype=float)));assert np.isnan(r.iloc[59]) and r.iloc[60]==1.
def test_block_stats():
 times=pd.date_range("2024-01-01",periods=1440,freq="1min",tz="UTC");rows=[]
 for j,symbol in enumerate(s.SYMBOLS):
  for i,t in enumerate(times):rows.append((t,symbol,100.,110+j,90.,101. if symbol=="BTCUSDT" else 100.))
 b=pd.DataFrame(rows,columns=["ts","symbol","open","high","low","close"]).set_index(["ts","symbol"]);ranges,ret,var=s.block_stats(b)
 assert len(ranges)==6 and all(v>0 for v in ranges.values()) and ret>0 and var>0
def test_active_controls():
 x=pd.DataFrame({"source_valid":[True]*3,"expansion_breadth":[4,3,5],"median_alt_range_rank":[.7,.8,.8],"btc_return":[.1,-.1,.2],"btc_variation_rank":[.8,.8,.4],"feature_available_time":pd.date_range("2024-01-01",periods=3,tz="UTC")})
 for c in s.RANK_COLS:x[c]=.8
 a,side,_=s.active(x);assert a.tolist()==[True,False,False] and side.tolist()==[1,-1,1]
 assert s.active(x,"three_of_six_breadth")[0].tolist()==[True,True,False]
 assert s.active(x,"forced_long")[1].tolist()==[1,1,1]
