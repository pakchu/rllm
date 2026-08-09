import numpy as np,pandas as pd
from training import build_lagged_flow_impact_coherence_relay_support as support
def _frame():
 t=pd.date_range(support.START,periods=96,freq="5min");flow=np.linspace(-.5,.5,96);q=np.full(96,100.);buy=.5*q*(1+flow);r=np.r_[0.,flow[:-1]/100];o=np.full(96,100.);c=o*np.exp(r);return pd.DataFrame({"bar_time":t,"bar_open":o,"bar_high":np.maximum(o,c),"bar_low":np.minimum(o,c),"bar_close":c,"quote_volume":q,"taker_buy_quote":buy,"source_rows":5,"distinct_rows":5,"first_ts":t,"last_ts":t+pd.Timedelta(minutes=4),"coherent":True})
def test_normalized_dot_and_rank():
 assert support.normalized_dot(np.array([1.,2.]),np.array([1.,2.]))==1.;x=pd.Series(range(361),dtype=float);r=support.strict_prior_midrank(x);assert np.isnan(r.iloc[359]) and r.iloc[360]==1.
def test_complete_feature_has_positive_lag():
 raw=_frame();old_start,old_end=support.START,support.END
 try:support.START=raw.bar_time.iloc[0];support.END=support.START+pd.Timedelta(hours=8);f=support.build_features(raw);assert f.iloc[0].source_valid and f.iloc[0].lagged_coherence>.99
 finally:support.START,support.END=old_start,old_end
def test_primary_onset_direction():
 f=pd.DataFrame({"source_valid":[1,1],"lagged_coherence":[.2,.2],"contemporaneous_coherence":[.1,.1],"coherence_rank":[.9,.9],"contemporaneous_rank":[.9,.9],"recent_flow":[.2,.2],"stale_recent_flow":[-.1,-.1],"recent_flow_magnitude_rank":[.8,.8],"primary_state":[1,1]});a,s=support.states_and_side(f);assert a.tolist()==[True,False] and s.tolist()==[1,1]
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
