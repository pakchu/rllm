import pandas as pd
from training import build_cboe_crypto_vix_transmission_relay_support as support
def frame():
 t=pd.Timestamp("2024-07-01T14:00:00Z");return pd.DataFrame({"observation_date":[pd.Timestamp("2024-06-28").date()],"next_source_date":[pd.Timestamp("2024-07-01").date()],"decision_time":[t],"delta_vix":[.02],"previous_delta_vix":[-.01],"bvol_body":[.03],"dvol_body":[.02],"valid":[True]})
def test_ccvtr_maps_confirmed_vix_rise_to_short():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==-1 and c.iloc[0].entry_time==pd.Timestamp("2024-07-01T14:05:00Z")
def test_ccvtr_requires_both_crypto_venues_to_confirm():
 f=frame();f.loc[0,"dvol_body"]=-.01;assert support.clock(f).empty and len(support.clock(f,"bvol_only_confirmation"))==1
def test_ccvtr_stale_and_flip_are_separate_controls():
 assert support.clock(frame(),"one_session_stale_vix_change").empty
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1 and a.iloc[0].side==-b.iloc[0].side
