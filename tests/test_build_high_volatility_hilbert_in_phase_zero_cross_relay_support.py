import numpy as np
import pandas as pd
from training import build_high_volatility_hilbert_in_phase_zero_cross_relay_support as support

CLOSE=[477.71,477.55,468.38,467.94,466.09,465.51,469.75,471.02,464.53,464.72,456.49,451.75,446.75,437.98,439.84,434.47,433.38,431.24,441.95,449.91]
IN_PHASE=[0,0,0,0,0,0,0,0,0,10.334625,-5.457375,-7.1865625,13.36986188,-2.887183125,2.728970313,9.788987291,-17.68573628,-12.36597885,-11.31680557,-42.62669254]
QUADRATURE=[0,0,0,0,0,0,0,2.26122,-2.28924,-11.01862764,-.56366312,-10.32737614,-8.189638135,-11.51239314,-23.99925769,-24.91090888,-27.5788291,-31.2253872,-31.25924424,-21.42226087]

def test_matches_official_lean_external_data():
 out=support.hilbert_components(pd.Series(CLOSE),pd.Series([True]*len(CLOSE)))
 assert np.allclose(out.in_phase,IN_PHASE,atol=1e-8) and np.allclose(out.quadrature,QUADRATURE,atol=1e-8)
 assert not out.in_phase_ready.iloc[8] and out.in_phase_ready.iloc[9]

def test_strict_in_phase_zero_cross_direction():
 out=support.hilbert_components(pd.Series(CLOSE),pd.Series([True]*len(CLOSE)))
 assert out.entry_side.iloc[10]==-1 and out.entry_side.iloc[12]==1 and out.entry_side.iloc[13]==-1

def test_invalid_bar_resets_all_delays_and_readiness():
 values=CLOSE+CLOSE;valid=pd.Series([True]*len(values));valid.iloc[20]=False
 out=support.hilbert_components(pd.Series(values),valid)
 assert not out.in_phase_ready.iloc[21:30].any() and out.in_phase_ready.iloc[30]

def test_prior_rank_excludes_current_and_resets():
 values=pd.Series(np.arange(125,dtype=float));valid=pd.Series([True]*125);rank=support.prior_rank(values,valid)
 assert np.isnan(rank.iloc[119]) and rank.iloc[120]==1.;valid.iloc[121]=False;assert np.isnan(support.prior_rank(values,valid).iloc[122])

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"quadrature_cross_side":[0,-1,0,1,0,0],"component_cross_side":[0,1,0,-1,0,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls_are_frozen_diagnostics():
 a,z,_=support.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert support.active(panel(),"no_variation_gate")[0].iloc[4]
 q,qs,_=support.active(panel(),"quadrature_zero_cross");assert q.iloc[1] and qs.iloc[1]==-1
 c,cs,_=support.active(panel(),"in_phase_quadrature_cross");assert c.iloc[3] and cs.iloc[3]==-1
 stale,ss,_=support.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and ss.iloc[3]==1
 flip,fs,_=support.active(panel(),"direction_flip");assert flip.iloc[2] and fs.iloc[2]==-1
 forced,fl,_=support.active(panel(),"forced_long");assert forced.iloc[2] and fl.iloc[2]==1
