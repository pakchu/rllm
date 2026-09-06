import numpy as np,pandas as pd
from training import build_range_body_efficiency_fracture_reversal_support as support
def _frame():
 t=pd.date_range(support.START,periods=96,freq="5min");o=np.full(96,100.);c=np.linspace(100.,101.,96);return pd.DataFrame({"bar_time":t,"bar_open":o,"bar_high":c+2,"bar_low":o-2,"bar_close":c,"source_rows":5,"distinct_rows":5,"first_ts":t,"last_ts":t+pd.Timedelta(minutes=4),"coherent":True})
def test_rank_and_crossing():
 x=pd.Series(range(721),dtype=float);r=support.strict_prior_midrank(x);assert np.isnan(r.iloc[719]) and r.iloc[720]==1.;assert support.crossing(pd.Series([.8,.9])).tolist()==[False,True]
def test_complete_feature():
 raw=_frame();old_start,old_end=support.START,support.END
 try:support.START=raw.bar_time.iloc[0];support.END=support.START+pd.Timedelta(hours=10);f=support.build_features(raw);assert f.iloc[0].source_valid and f.iloc[0].fracture>0 and f.iloc[0].impulse>0
 finally:support.START,support.END=old_start,old_end
def test_direction_flip():
 f=pd.DataFrame({"source_valid":[1,1],"fracture_rank":[.8,.9],"raw_range_rank":[.8,.9],"inverse_body_rank":[.8,.9],"impulse":[.1,.1]});a,s=support.active_and_side(f);assert a.tolist()==[False,True] and s.tolist()==[-1,-1];_,flip=support.active_and_side(f,"direction_flip");assert flip.tolist()==[1,1]
def test_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s and '"gross9_rows_opened":False' in s and '"rv20_opened":False' in s
