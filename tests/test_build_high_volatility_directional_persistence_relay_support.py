import pandas as pd
from training import build_high_volatility_directional_persistence_relay_support as support
def test_hvdpr_clock_enters_in_persistent_direction():
 d=pd.Timestamp("2024-07-01T08:00:00Z");f=pd.DataFrame({"decision_time":[d],"block_valid":[True],"bvol_close":[70.],"prior_bvol_q60":[60.],"dvol_close":[70.],"prior_dvol_q60":[60.],"first_half_return":[.02],"second_half_return":[.03],"block_return":[.05],"prior_abs_block_q60":[.03],"path_efficiency":[.8],"prior_efficiency_q60":[.6]});c=support.clock(f);assert len(c)==1 and c.iloc[0].side==1 and c.iloc[0].entry_time==d+pd.Timedelta(minutes=5)
def test_hvdpr_rejects_opposed_halves_or_low_efficiency():
 d=pd.Timestamp("2024-07-01T08:00:00Z");common={"decision_time":d,"block_valid":True,"bvol_close":70.,"prior_bvol_q60":60.,"dvol_close":70.,"prior_dvol_q60":60.,"first_half_return":.02,"second_half_return":.03,"block_return":.05,"prior_abs_block_q60":.03,"path_efficiency":.8,"prior_efficiency_q60":.6};assert support.clock(pd.DataFrame([{**common,"second_half_return":-.03}])).empty;assert support.clock(pd.DataFrame([{**common,"path_efficiency":.5}])).empty
