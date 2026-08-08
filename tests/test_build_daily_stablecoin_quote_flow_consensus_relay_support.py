import pandas as pd
from training import build_daily_stablecoin_quote_flow_consensus_relay_support as support

def frame():
 d=pd.Timestamp("2024-07-02T00:00:00Z");return pd.DataFrame({"source_day":[d-pd.Timedelta(days=1)],"decision_time":[d],"flow_valid":[True],"vol_valid":[True],"flow_usdt":[.08],"flow_usdc":[.06],"usdc_volume_share":[.02],"bvol_close":[70.],"prior_bvol_q60":[60.],"dvol_close":[68.],"prior_dvol_q60":[60.]})
def test_dsqfcr_follows_two_book_daily_consensus():
 c=support.clock(frame());assert len(c)==1;assert c.iloc[0].side==1;assert c.iloc[0].entry_time==pd.Timestamp("2024-07-02T00:05:00Z");assert c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=12)
def test_dsqfcr_rejects_disagreement_low_flow_or_low_vol():
 x=frame();x.loc[0,"flow_usdc"]=-.06;assert support.clock(x).empty;x=frame();x.loc[0,"flow_usdc"]=.04;assert support.clock(x).empty;x=frame();x.loc[0,"dvol_close"]=50.;assert support.clock(x).empty
def test_dsqfcr_direction_flip_is_diagnostic_only():
 a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert len(a)==len(b)==1;assert a.iloc[0].side==-b.iloc[0].side;assert b.iloc[0].control=="direction_flip"

def test_dsqfcr_bound_sources_build_numeric_daily_features():
 f=support.features();assert not f.empty;assert f[["flow_usdt","flow_usdc","usdc_volume_share"]].dtypes.apply(lambda x:x.kind in "fc").all()
