import numpy as np
import pandas as pd
from training import build_high_volatility_larry_williams_range_breakout_relay_support as s

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1
def panel():return pd.DataFrame({"source_valid":[True]*7,"hour_close":[10.,13.,12.,7.,13.,12.,7.],"previous_reference":[10.,10.,13.,12.,7.,13.,12.],"upper":[12.]*7,"lower":[8.]*7,"body_upper":[11.]*7,"body_lower":[9.]*7,"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[1] and z.iloc[1]==1 and a.iloc[3] and z.iloc[3]==-1 and not a.iloc[4] and a.iloc[6] and z.iloc[6]==-1
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 body,side,_=s.active(panel(),"prior_day_body_projection");assert body.iloc[1] and side.iloc[1]==1 and body.iloc[3] and side.iloc[3]==-1
 stale,side,_=s.active(panel(),"one_hour_stale_crossing");assert stale.iloc[2] and side.iloc[2]==1
 flipped,side,_=s.active(panel(),"direction_flip");assert flipped.iloc[1] and side.iloc[1]==-1
