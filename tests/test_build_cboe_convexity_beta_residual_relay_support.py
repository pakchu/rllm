import numpy as np,pandas as pd
from training import build_cboe_convexity_beta_residual_relay_support as support
def frame():
 d=pd.to_datetime(["2024-07-01","2024-07-02","2024-07-03"]);return pd.DataFrame({"observation_date":d,"delta_log_vix":[.01,.02,.01],"delta_log_vvix":[.02,.08,.02],"rolling_intercept":[0.,0.,0.],"rolling_beta":[2.,2.,2.],"standardized_residual":[0.,1.5,0.],"raw_vvix_z":[0.,1.5,0.],"fixed_beta_one_z":[0.,1.5,0.]})
def test_ccbrr_positive_convexity_residual_maps_short():
 c=support.clock(frame());assert len(c)==1;assert c.iloc[0].side==-1;assert c.iloc[0].entry_time==pd.Timestamp("2024-07-03T13:35:00Z");assert c.iloc[0].exit_time-c.iloc[0].entry_time==pd.Timedelta(hours=12)
def test_ccbrr_magnitude_gate_and_direction_flip():
 f=frame();f.loc[1,"standardized_residual"]=.9;assert support.clock(f).empty;f=frame();a=support.clock(f);b=support.clock(f,"direction_flip");assert a.iloc[0].side==-b.iloc[0].side
def test_causal_residual_excludes_current():
 x=pd.Series(np.linspace(-1,1,127));y=2*x;y.iloc[-1]+=1.;r=support.causal_residual(x,y,minimum=126);assert r.standardized_residual.iloc[:126].isna().all();assert r.standardized_residual.iloc[126]>1e6
