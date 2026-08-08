import pandas as pd
from training import build_cross_venue_volatility_ignition_breakout_relay_support as support

def test_cvvib_clock_enters_after_completed_four_hour_block():
    decision=pd.Timestamp("2024-07-01T08:00:00Z")
    frame=pd.DataFrame({"decision_time":[decision],"block_valid":[True],"bvol_block_return":[.1],"dvol_block_return":[.08],"bvol_body":[.02],"dvol_body":[.01],"first_three_hour_return":[.001],"prior_abs_first_three_q50":[.002],"fourth_hour_return":[-.01],"prior_abs_fourth_q60":[.005]})
    clock=support.build_clock(frame)
    assert len(clock)==1 and clock.iloc[0].side==-1
    assert clock.iloc[0].entry_time==decision+pd.Timedelta(minutes=5)
    assert clock.iloc[0].exit_time-clock.iloc[0].entry_time==pd.Timedelta(hours=6)

def test_cvvib_primary_requires_joint_late_volatility_ignition():
    decision=pd.Timestamp("2024-07-01T08:00:00Z")
    common={"decision_time":decision,"block_valid":True,"bvol_block_return":.1,"dvol_block_return":.08,"bvol_body":.02,"dvol_body":.01,"first_three_hour_return":.001,"prior_abs_first_three_q50":.002,"fourth_hour_return":.01,"prior_abs_fourth_q60":.005}
    assert support.build_clock(pd.DataFrame([{**common,"dvol_block_return":-.01}])).empty
    assert support.build_clock(pd.DataFrame([{**common,"dvol_body":-.01}])).empty
