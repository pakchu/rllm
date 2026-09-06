import numpy as np,pandas as pd
from training import build_high_volatility_price_impact_asymmetry_relay_support as support
def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(253,dtype=float)));assert r.iloc[:252].isna().all();assert r.iloc[252]==1.
def _states(**change):
 row={"block_start":pd.Timestamp("2023-07-01T00:00:00Z"),"decision_time":pd.Timestamp("2023-07-01T08:00:00Z"),"source_valid":True,"impact_asymmetry":.1,"asymmetry_rank":.9,"variation":.1,"variation_rank":.8,"positive_bars":40,"negative_bars":40,"upside_impact":.001,"downside_impact":.002};row.update(change);return pd.DataFrame([row])
def test_primary_relays_toward_fragile_side():
 s=_states();c=support.build_clock(s);assert c.side.tolist()==[-1];assert support.build_clock(s,"direction_flip").side.tolist()==[1]
def test_asymmetry_tail_is_required_only_by_control():
 s=_states(asymmetry_rank=.2);assert support.build_clock(s).empty;assert support.build_clock(s,"no_asymmetry_tail").side.tolist()==[-1]
