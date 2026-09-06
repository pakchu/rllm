import pandas as pd
from training import build_joint_volatility_cooling_trend_relay_support as support
def test_jvctr_continues_trend_when_both_volatility_indices_cool():
 d=pd.Timestamp("2024-07-01T08:00:00Z");f=pd.DataFrame({"decision_time":[d],"four_valid":[True],"bvol_body":[-.01],"dvol_body":[-.02],"trend_return_3h":[.03],"prior_abs_trend_q60":[.02],"pullback_return_1h":[.01]});c=support.clock(f);assert len(c)==1 and c.iloc[0].side==1 and c.iloc[0].entry_time==d+pd.Timedelta(minutes=5)
def test_jvctr_rejects_expanding_volatility_or_price_reversal():
 d=pd.Timestamp("2024-07-01T08:00:00Z");common={"decision_time":d,"four_valid":True,"bvol_body":-.01,"dvol_body":-.02,"trend_return_3h":.03,"prior_abs_trend_q60":.02,"pullback_return_1h":.01};assert support.clock(pd.DataFrame([{**common,"dvol_body":.02}])).empty;assert support.clock(pd.DataFrame([{**common,"pullback_return_1h":-.01}])).empty
