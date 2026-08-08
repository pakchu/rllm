import pandas as pd
from training import build_dominant_quote_deleveraging_ignition_relay_support as support
def frame():
 t=pd.Timestamp("2024-07-01T08:00:00Z");return pd.DataFrame({"source_hour_start":[t-pd.Timedelta(hours=2),t-pd.Timedelta(hours=1)],"decision_time":[t-pd.Timedelta(hours=1),t],"signal_valid":[True,True],"z_usdt":[0.,-1.],"z_usdc":[0.,.2],"z_fdusd":[0.,-.3],"oi_change":[0.,-.01],"oi_current_time":[t-pd.Timedelta(hours=1),t],"oi_prior_time":[t-pd.Timedelta(hours=2),t-pd.Timedelta(hours=1)],"bvol_body":[-.01,.02],"dvol_body":[-.01,.03]})
def test_dqdir_follows_dominant_usdt_flow_during_deleveraging_ignition():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==-1 and c.iloc[0].entry_time==pd.Timestamp("2024-07-01T08:05:00Z")
def test_dqdir_rejects_active_alternative_book_oi_build_or_single_vol_expansion():
 x=frame();x.loc[1,"z_usdc"]=.5;assert support.clock(x).empty
 x=frame();x.loc[1,"oi_change"]=.01;assert support.clock(x).empty
 x=frame();x.loc[1,"dvol_body"]=-.01;assert support.clock(x).empty
def test_dqdir_direction_flip_preserves_clock():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1 and a.iloc[0].side==-b.iloc[0].side and a.iloc[0].entry_time==b.iloc[0].entry_time
