import numpy as np,pandas as pd
from training import build_high_volatility_leverage_to_cash_flow_handoff_relay_support as s
def test_rank_excludes_current():
 r=s.rank(pd.Series(np.arange(181,dtype=float)));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def _frame(**u):
 rows=[]
 for i in range(2):
  x={"decision_time":pd.Timestamp("2023-07-01T03:00:00Z")+pd.Timedelta(hours=8*i),"source_valid":True,"ordered_handoff":True,"side":1,"perp_first_flow":.2,"spot_first_flow":.1,"perp_second_flow":.1,"spot_second_flow":.2,"handoff_strength":.1,"handoff_rank":.8,"variation":.1,"variation_rank":.8};x.update(u);rows.append(x)
 return pd.DataFrame(rows)
def test_onset_and_side():
 x=_frame();x.loc[0,"handoff_rank"]=.2;c=s.clock(x);assert c.side.tolist()==[1];assert c.decision_time.tolist()==[pd.Timestamp("2023-07-01T11:00:00Z")];assert len(s.clock(x,"no_onset"))==1
def test_controls():
 x=_frame(handoff_rank=.2);x.loc[0,"ordered_handoff"]=False;assert s.clock(x).empty;assert len(s.clock(x,"no_handoff_tail"))==1
 x=_frame(variation_rank=.2);x.loc[0,"handoff_rank"]=.2;assert s.clock(x).empty;assert s.clock(x,"no_variation_gate").side.tolist()==[1]
