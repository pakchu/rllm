import pandas as pd
from training import build_oi_divergence_contradiction_relay_support as support
def frame():return pd.DataFrame({"date":pd.date_range("2023-07-01T00:25:00Z",periods=2,freq="30min"),"oi_minus_px_4h_z":[1.,-1.],"return_zscore_48":[-1.,1.],"range_vol":[.05,.05],"rsi_norm":[-.1,.1]})
def test_oidcr_mirrors_inventory_contradiction():
 lo,sh=support.signals(frame());assert lo.tolist()==[True,False] and sh.tolist()==[False,True]
def test_oidcr_global_reservation_keeps_first_state():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==1
