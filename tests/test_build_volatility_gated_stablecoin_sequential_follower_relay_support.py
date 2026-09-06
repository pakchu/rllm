import pandas as pd
from training import build_volatility_gated_stablecoin_sequential_follower_relay_support as support

def frame(current_usdc=.6,current_fdusd=.8):
 t=pd.Timestamp("2024-07-01T08:00:00Z")
 return pd.DataFrame({"source_hour_start":[t-pd.Timedelta(hours=2),t-pd.Timedelta(hours=1)],"decision_time":[t-pd.Timedelta(hours=1),t],"source_valid":[True,True],"vol_valid":[True,True],"z_usdt":[0.,0.],"z_usdc":[1.2,current_usdc],"z_fdusd":[.2,current_fdusd],"alt_share":[.6,.6],"prior_alt_share_q50":[.5,.5],"bvol_close":[70.,70.],"prior_bvol_q60":[60.,60.],"dvol_close":[65.,65.],"prior_dvol_q60":[60.,60.]})
def test_vgsfr_clock_follows_prior_usdc_leader_after_current_fdusd_confirmation():
 c=support.clock(frame());assert len(c)==1 and c.iloc[0].side==1 and c.iloc[0].leader=="BTCUSDC"
 assert c.iloc[0].entry_time==pd.Timestamp("2024-07-01T08:05:00Z") and c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=6)
def test_vgsfr_rejects_weak_follower_or_low_volatility():
 weak=frame(current_fdusd=.7)
 assert support.clock(weak).empty
 low=frame();low.loc[1,"dvol_close"]=50.
 assert support.clock(low).empty
