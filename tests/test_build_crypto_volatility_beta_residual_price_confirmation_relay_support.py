import pandas as pd
from training import build_crypto_volatility_beta_residual_price_confirmation_relay_support as support
def frame():
 d=pd.Timestamp("2024-07-02T00:00:00Z");return pd.DataFrame({"decision_time":[d],"source_valid":[True],"delta_log_bvol":[.02],"delta_log_dvol":[.05],"rolling_intercept":[0.],"rolling_beta":[1.],"standardized_residual":[1.5],"raw_dvol_z":[1.5],"fixed_beta_one_z":[1.5],"price_return_24h":[-.03]})
def test_cvbrpcr_uses_price_confirmation_direction():
 c=support.clock(frame());assert len(c)==1;assert c.iloc[0].side==-1;assert c.iloc[0].entry_time==pd.Timestamp("2024-07-02T00:05:00Z");assert c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=12)
def test_cvbrpcr_residual_gate_and_direction_flip():
 x=frame();x.loc[0,"standardized_residual"]=.9;assert support.clock(x).empty;a=support.clock(frame());b=support.clock(frame(),"direction_flip");assert a.iloc[0].side==-b.iloc[0].side
def test_cvbrpcr_residual_direction_is_diagnostic():
 c=support.clock(frame(),"residual_direction");assert len(c)==1;assert c.iloc[0].side==-1
