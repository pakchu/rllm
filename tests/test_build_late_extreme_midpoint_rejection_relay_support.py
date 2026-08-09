import numpy as np,pandas as pd
from training import build_late_extreme_midpoint_rejection_relay_support as support
def _frame():
 t=pd.date_range("2024-07-08",periods=1440,freq="1min",tz="UTC");o=np.full(1440,100.);c=np.full(1440,100.);h=np.full(1440,101.);l=np.full(1440,99.);l[100]=80.;h[1000]=120.;c[-1]=90.;return pd.DataFrame({"ts":t,"open":o,"high":np.maximum(h,np.maximum(o,c)),"low":np.minimum(l,np.minimum(o,c)),"close":c})
def test_late_high_rejection_and_clock():
 f=support.build_features(_frame());assert f.iloc[0].late_high_rejection and not f.iloc[0].late_low_rejection;assert support.signal(f).iloc[0]==-1;c=support.build_clock(f);assert c.iloc[0].entry_time==pd.Timestamp("2024-07-09T00:05Z") and c.iloc[0].exit_time==pd.Timestamp("2024-07-09T16:05Z")
def test_missing_row_fails_closed():assert not support.build_features(_frame().drop(index=5)).iloc[0].source_valid
def test_controls_and_gates():
 f=support.build_features(_frame());assert support.signal(f,"no_midpoint_rejection").iloc[0]==-1;assert support.signal(f,"direction_flip").iloc[0]==1;assert support.MINIMUM=={"train":8,"test":12,"eval":12,"final":8}
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
