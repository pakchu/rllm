import numpy as np
import pandas as pd
from training import build_high_volatility_cross_alt_volume_weighted_auction_consensus_relay_support as s

def block(displacements):
 rows=[];times=pd.date_range("2025-01-01",periods=480,freq="1min",tz="UTC")
 for symbol,move in zip(s.ALTS,displacements):
  close=np.ones(480);close[-1]=np.exp(move)
  for t,c in zip(times,close):rows.append((t,symbol,c,c,c,c,1.))
 for t in times:rows.append((t,"BTCUSDT",1.,1.,1.,np.exp(.001),1.))
 return pd.DataFrame(rows,columns=["ts","symbol","open","high","low","close","quote_asset_volume"]).set_index(["ts","symbol"]).sort_index()

def test_consensus_uses_volume_weighted_value_displacement():
 side,strength,equal_side,breadth,variation=s.auction_value_consensus_statistics(block([.2,.2,.2,.2,-.2,-.2]))
 assert side==1 and breadth==4 and strength>0 and variation>0
 assert equal_side==1

def test_three_three_tie_is_not_consensus():
 side,strength,_,breadth,_=s.auction_value_consensus_statistics(block([.2,.2,.2,-.2,-.2,-.2]))
 assert side==0 and breadth==3 and np.isnan(strength)

def test_pinned_blind_registration():
 assert s.PREREG_SHA=="4624e40689177633a7bd9f188636bf903a4c3cf3669b23643e2e33cb1fd5f321"
