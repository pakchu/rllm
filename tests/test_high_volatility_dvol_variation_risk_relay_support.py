import numpy as np,pandas as pd
from training import build_high_volatility_dvol_variation_risk_relay_support as support
def test_rank_excludes_current_and_matures_at_336():
 x=pd.Series(np.arange(337,dtype=float));r=support.rank(x);assert r.iloc[:336].isna().all();assert r.iloc[336]==1.
def _states(**change):
 row={"decision_time":pd.Timestamp("2023-07-02T00:00:00Z"),"state_valid":True,"dvol_change":.1,"dvol_variation":.2,"dvol_variation_rank":.8,"btc_variation":.1,"btc_variation_rank":.8};row.update(change);return pd.DataFrame([row])
def test_primary_maps_rising_dvol_short_and_controls_are_diagnostic():
 s=_states();assert support.build_clock(s).side.tolist()==[-1];assert support.build_clock(s,"dvol_direction").side.tolist()==[1];assert support.build_clock(s,"direction_flip").side.tolist()==[1];assert support.build_clock(s,"same_clock_forced_long").side.tolist()==[1]
def test_each_tail_gate_is_mandatory_only_for_primary():
 s=_states(dvol_variation_rank=.2);assert support.build_clock(s).empty;assert support.build_clock(s,"no_dvol_variation_gate").side.tolist()==[-1]
