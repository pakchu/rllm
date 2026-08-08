import pandas as pd
from training import build_joint_volatility_intrahour_acceleration_relay_support as support

def frame():
 t=pd.Timestamp("2024-07-01T08:00:00Z")
 return pd.DataFrame({"decision_time":[t-pd.Timedelta(hours=1),t],"signal_valid":[True,True],"bvol_body":[-0.01,.02],"dvol_body":[-0.01,.03],"first_half_return":[.001,.02],"second_half_return":[-.001,.03],"prior_abs_first_half_q60":[.01,.01]})
def test_jviar_follows_same_direction_intrahour_acceleration():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==1
 assert c.iloc[0].entry_time==pd.Timestamp("2024-07-01T08:05:00Z") and c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=6)
def test_jviar_rejects_absorption_deceleration_and_single_venue_expansion():
 x=frame();x.loc[1,"second_half_return"]=-.03;assert support.clock(x).empty
 x=frame();x.loc[1,"second_half_return"]=.01;assert support.clock(x).empty
 x=frame();x.loc[1,"dvol_body"]=-.01;assert support.clock(x).empty
def test_jviar_direction_flip_is_clock_identical():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1 and a.iloc[0].side==-b.iloc[0].side and a.iloc[0].entry_time==b.iloc[0].entry_time
