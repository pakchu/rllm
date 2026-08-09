import numpy as np,pandas as pd
from training import build_weekend_monday_liquidity_handoff_relay_support as support
def _frame():
 t=pd.date_range("2024-07-06",periods=3360,freq="1min",tz="UTC");o=np.where(t<pd.Timestamp("2024-07-08",tz="UTC"),100.,90.);c=np.where(t<pd.Timestamp("2024-07-08",tz="UTC"),90.,100.);q=np.where(t<pd.Timestamp("2024-07-08",tz="UTC"),1.,10.);return pd.DataFrame({"ts":t,"open":o,"high":np.maximum(o,c),"low":np.minimum(o,c),"close":c,"quote_asset_volume":q})
def test_handoff_feature_and_clock():
 f=support.build_features(_frame());assert f.iloc[0].direction_handoff and f.iloc[0].liquidity_reentry;assert support.signal(f).iloc[0]==1;c=support.build_clock(f);assert c.iloc[0].entry_time==pd.Timestamp("2024-07-08T08:05Z") and c.iloc[0].exit_time==pd.Timestamp("2024-07-09T00:05Z")
def test_missing_row_fails_closed():assert not support.build_features(_frame().drop(index=5)).iloc[0].source_valid
def test_controls_and_gates():
 f=support.build_features(_frame());assert support.signal(f,"weekend_continuation").iloc[0]==-1;assert support.signal(f,"direction_flip").iloc[0]==-1;assert support.MINIMUM=={"train":8,"test":12,"eval":12,"final":8}
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
