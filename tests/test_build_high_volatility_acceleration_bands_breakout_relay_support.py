import numpy as np
import pandas as pd
from training import build_high_volatility_acceleration_bands_breakout_relay_support as support

def bars(values):
 return pd.DataFrame(values,columns=["bar_high","bar_low","bar_close"])

def test_official_width_four_transformation_and_sma():
 values=[(101.,99.,100.)]*10
 out=support.acceleration_bands(bars(values),pd.Series([True]*10))
 assert np.isclose(out.coefficient.iloc[-1],.04) and np.isclose(out.raw_upper.iloc[-1],105.04) and np.isclose(out.raw_lower.iloc[-1],95.04)
 assert out.bands_ready.iloc[-1] and np.isclose(out.upper_band.iloc[-1],105.04) and np.isclose(out.middle_band.iloc[-1],100.)

def test_outward_upper_cross_is_long():
 values=[(101.,99.,100.)]*10+[(110.,100.,110.)]
 out=support.acceleration_bands(bars(values),pd.Series([True]*11))
 assert out.entry_side.iloc[-1]==1 and out.middle_cross_side.iloc[-1]==1

def test_outward_lower_cross_is_short():
 values=[(101.,99.,100.)]*10+[(100.,90.,90.)]
 out=support.acceleration_bands(bars(values),pd.Series([True]*11))
 assert out.entry_side.iloc[-1]==-1

def test_raw_transformed_boundary_cannot_be_breached_by_valid_close():
 values=[(101.,99.,100.)]*10+[(110.,100.,110.),(100.,90.,90.)]
 out=support.acceleration_bands(bars(values),pd.Series([True]*12))
 assert not out.raw_boundary_side.ne(0).any()

def test_invalid_bar_resets_ten_bar_readiness_and_cross_state():
 values=[(101.,99.,100.)]*22;valid=pd.Series([True]*11+[False]+[True]*10)
 out=support.acceleration_bands(bars(values),valid)
 assert not out.bands_ready.iloc[12:21].any() and out.bands_ready.iloc[21]

def test_prior_rank_excludes_current_and_resets():
 values=pd.Series(np.arange(125,dtype=float));valid=pd.Series([True]*125);rank=support.prior_rank(values,valid)
 assert np.isnan(rank.iloc[119]) and rank.iloc[120]==1.;valid.iloc[121]=False;assert np.isnan(support.prior_rank(values,valid).iloc[122])

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"middle_cross_side":[0,-1,0,1,0,0],"raw_boundary_side":[0,0,0,0,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls_are_frozen_diagnostics():
 a,z,_=support.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert support.active(panel(),"no_variation_gate")[0].iloc[4]
 m,ms,_=support.active(panel(),"middle_band_cross");assert m.iloc[1] and ms.iloc[1]==-1
 assert not support.active(panel(),"raw_transformed_boundary_break")[0].any()
 stale,ss,_=support.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and ss.iloc[3]==1
 flip,fs,_=support.active(panel(),"direction_flip");assert flip.iloc[2] and fs.iloc[2]==-1
 forced,fl,_=support.active(panel(),"forced_long");assert forced.iloc[2] and fl.iloc[2]==1
