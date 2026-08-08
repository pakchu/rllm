import pandas as pd
from training import build_spot_participation_volatility_ignition_relay_support as support
def frame():
 t=pd.Timestamp("2024-07-01T08:00:00Z");return pd.DataFrame({"decision_time":[t-pd.Timedelta(hours=1),t],"signal_valid":[True,True],"spot_return":[-.01,.02],"perpetual_return":[.01,.018],"spot_participation":[.2,.4],"prior_participation_q75":[.3,.3],"bvol_body":[-.01,.02],"dvol_body":[-.01,.03]})
def test_spvir_relays_same_direction_cash_participation_ignition():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==1 and c.iloc[0].entry_time==pd.Timestamp("2024-07-01T08:05:00Z")
def test_spvir_rejects_low_participation_disagreement_and_single_venue_expansion():
 x=frame();x.loc[1,"spot_participation"]=.2;assert support.clock(x).empty
 x=frame();x.loc[1,"perpetual_return"]=-.018;assert support.clock(x).empty
 x=frame();x.loc[1,"dvol_body"]=-.01;assert support.clock(x).empty
def test_spvir_direction_flip_is_clock_identical():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1 and a.iloc[0].entry_time==b.iloc[0].entry_time and a.iloc[0].side==-b.iloc[0].side
