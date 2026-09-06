import numpy as np,pandas as pd
from training import build_high_volatility_kurtosis_regime_momentum_relay_support as support
def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(181,dtype=float)));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def _states(**change):
 row={"source_day":pd.Timestamp("2023-07-01T00:00:00Z"),"decision_time":pd.Timestamp("2023-07-02T00:00:00Z"),"source_valid":True,"day_return":.1,"realized_variation":.01,"realized_kurtosis":9.,"kurtosis_rank":.9,"variation_rank":.8};row.update(change);return pd.DataFrame([row])
def test_primary_follows_day_and_controls_are_diagnostic():
 s=_states();c=support.build_clock(s);assert c.side.tolist()==[1];assert c.entry_time.iloc[0]==pd.Timestamp("2023-07-02T00:05:00Z");assert support.build_clock(s,"direction_flip").side.tolist()==[-1]
def test_kurtosis_tail_is_required_only_by_control():
 s=_states(kurtosis_rank=.2);assert support.build_clock(s).empty;assert support.build_clock(s,"no_kurtosis_tail").side.tolist()==[1]
