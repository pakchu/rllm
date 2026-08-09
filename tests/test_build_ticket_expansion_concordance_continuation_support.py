import numpy as np,pandas as pd
from training import build_ticket_expansion_concordance_continuation_support as support
def _frame():
 t=pd.date_range(support.START,periods=480,freq="1min");o=np.full(480,100.);c=np.r_[np.full(360,105.),np.full(120,110.)];q=np.r_[np.full(360,100.),np.full(120,200.)];n=np.full(480,10.);return pd.DataFrame({"ts":t,"open":o,"high":np.maximum(o,c),"low":np.minimum(o,c),"close":c,"quote_asset_volume":q,"number_of_trades":n})
def test_strict_prior_rank_excludes_current():
 x=pd.Series(range(361),dtype=float);r=support.strict_prior_midrank(x,540,360);assert r.iloc[359]!=r.iloc[359] and r.iloc[360]==1.0
def test_complete_feature_and_state():
 raw=_frame();old_start,old_end=support.START,support.END
 try:
  support.START=raw.ts.iloc[0];support.END=support.START+pd.Timedelta(hours=8);f=support.build_features(raw);assert f.iloc[0].source_valid and f.iloc[0].concordant and f.iloc[0].ticket_expansion>0
 finally:support.START,support.END=old_start,old_end
def test_onset_and_direction():
 f=pd.DataFrame({"source_valid":[1,1,1],"concordant":[1,1,1],"ticket_expansion":[1.,1.,-1.],"ticket_expansion_rank":[.9,.9,.1],"early_dominance_rank":[.1,.1,.9],"late_return":[.1,.1,-.1],"primary_state":[1,1,0]});a,s=support.states_and_side(f);assert a.tolist()==[True,False,False] and s.tolist()==[1,1,-1]
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
