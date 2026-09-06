import pandas as pd
from training import build_high_volatility_trend_pullback_relay_support as support
def test_hvtpr_continues_original_trend_after_shallow_pullback():
 d=pd.Timestamp("2024-07-01T08:00:00Z");f=pd.DataFrame({"decision_time":[d],"four_valid":[True],"bvol_close":[70.],"prior_bvol_q60":[60.],"dvol_close":[70.],"prior_dvol_q60":[60.],"trend_return_3h":[.03],"prior_abs_trend_q60":[.02],"pullback_return_1h":[-.01]});c=support.clock(f);assert len(c)==1 and c.iloc[0].side==1 and c.iloc[0].entry_time==d+pd.Timedelta(minutes=5)
def test_hvtpr_rejects_deep_or_same_direction_hour():
 d=pd.Timestamp("2024-07-01T08:00:00Z");common={"decision_time":d,"four_valid":True,"bvol_close":70.,"prior_bvol_q60":60.,"dvol_close":70.,"prior_dvol_q60":60.,"trend_return_3h":.03,"prior_abs_trend_q60":.02,"pullback_return_1h":-.01};assert support.clock(pd.DataFrame([{**common,"pullback_return_1h":-.02}])).empty;assert support.clock(pd.DataFrame([{**common,"pullback_return_1h":.01}])).empty
