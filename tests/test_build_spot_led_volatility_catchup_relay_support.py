import pandas as pd
from training import build_spot_led_volatility_catchup_relay_support as support
def frame():
 t=pd.Timestamp("2024-07-01T08:00:00Z");return pd.DataFrame({"decision_time":[t-pd.Timedelta(hours=1),t],"signal_valid":[True,True],"spot_return":[-.001,.02],"perpetual_return":[.001,.008],"prior_abs_spot_q60":[.01,.01],"bvol_body":[-.01,.02],"dvol_body":[-.01,.03]})
def test_slvcr_relays_cash_led_partial_perpetual_transmission():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==1 and c.iloc[0].entry_time==pd.Timestamp("2024-07-01T08:05:00Z")
def test_slvcr_rejects_disagreement_full_transmission_and_single_venue_volatility():
 x=frame();x.loc[1,"perpetual_return"]=-.008;assert support.clock(x).empty
 x=frame();x.loc[1,"perpetual_return"]=.015;assert support.clock(x).empty
 x=frame();x.loc[1,"dvol_body"]=-.01;assert support.clock(x).empty
def test_slvcr_direction_flip_is_clock_identical():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1 and a.iloc[0].entry_time==b.iloc[0].entry_time and a.iloc[0].side==-b.iloc[0].side
