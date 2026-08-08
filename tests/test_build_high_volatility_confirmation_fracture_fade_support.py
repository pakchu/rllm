import pandas as pd
from training import build_high_volatility_confirmation_fracture_fade_support as support
def test_hvcff_source_validity_does_not_depend_on_oi_or_funding_columns():
 source=support.features();assert source.prior_fracture_q60.notna().sum()>100;assert source.prior_abs_block_q60.notna().sum()>1000;assert source.block_valid.sum()>1000
def test_hvcff_fades_completed_large_move():
 d=pd.Timestamp("2024-07-01T08:00:00Z");f=pd.DataFrame({"decision_time":[d],"block_valid":[True],"bvol_close":[80.],"prior_bvol_q60":[60.],"prior_bvol_median":[55.],"dvol_close":[70.],"prior_dvol_q60":[60.],"prior_dvol_median":[60.],"normalized_level_fracture":[.2],"prior_fracture_q60":[.1],"block_return":[.03],"prior_abs_block_q60":[.02]});c=support.clock(f);assert len(c)==1 and c.iloc[0].side==-1 and c.iloc[0].entry_time==d+pd.Timedelta(minutes=5)
def test_hvcff_requires_level_fracture_and_large_move():
 d=pd.Timestamp("2024-07-01T08:00:00Z");common={"decision_time":d,"block_valid":True,"bvol_close":80.,"prior_bvol_q60":60.,"prior_bvol_median":55.,"dvol_close":70.,"prior_dvol_q60":60.,"prior_dvol_median":60.,"normalized_level_fracture":.2,"prior_fracture_q60":.1,"block_return":.03,"prior_abs_block_q60":.02};assert support.clock(pd.DataFrame([{**common,"normalized_level_fracture":.05}])).empty;assert support.clock(pd.DataFrame([{**common,"block_return":.01}])).empty
