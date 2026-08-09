import numpy as np,pandas as pd
from training import build_asian_session_concordance_relay_support as support
def _frame():
 t=pd.date_range("2024-07-01",periods=2,freq="4h",tz="UTC");return pd.DataFrame({"block_start":t,"block_open":[100.,105.],"block_high":[105.,110.],"block_low":[100.,105.],"block_close":[105.,110.],"source_rows":240,"distinct_rows":240,"first_ts":t,"last_ts":t+pd.Timedelta(minutes=239),"coherent":True})
def test_concordance_and_clock():
 old_start,old_end=support.START,support.END
 try:support.START=pd.Timestamp('2024-07-01',tz='UTC');support.END=support.START+pd.Timedelta(days=1);f=support.build_features(_frame());assert f.iloc[0].concordant and support.signal(f).iloc[0]==1;c=support.build_clock(f);assert c.iloc[0].entry_time==pd.Timestamp('2024-07-01T08:05Z') and c.iloc[0].exit_time==pd.Timestamp('2024-07-02T00:05Z')
 finally:support.START,support.END=old_start,old_end
def test_missing_block_fails_closed():
 old_start,old_end=support.START,support.END
 try:support.START=pd.Timestamp('2024-07-01',tz='UTC');support.END=support.START+pd.Timedelta(days=1);assert not support.build_features(_frame().iloc[:1]).iloc[0].source_valid
 finally:support.START,support.END=old_start,old_end
def test_controls_and_gates():
 assert support.MINIMUM=={"train":8,"test":12,"eval":12,"final":8}
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
