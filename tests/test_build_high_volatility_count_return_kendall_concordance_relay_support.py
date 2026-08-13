import numpy as np,pandas as pd,pytest
from training import build_high_volatility_count_return_kendall_concordance_relay_support as s

def test_tau_b_concordant_discordant_and_ties():
 x=np.arange(96,dtype=float);assert s.kendall_tau_b(x,x)==pytest.approx(1.);assert s.kendall_tau_b(x,-x)==pytest.approx(-1.)
 y=x.copy();x[:2]=0;y[:2]=0;assert np.isfinite(s.kendall_tau_b(x,y))
def test_tau_b_fails_closed():assert np.isnan(s.kendall_tau_b(np.arange(95),np.arange(95)))
def test_rank_excludes_current():
 r=s.rank(pd.Series(range(181),dtype=float));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def test_onset_side_controls_and_prereg_hash():
 d=pd.date_range("2024-07-01T02:00:00Z",periods=4,freq="8h");x=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"kendall_tau_b":[.2]*4,"pearson_correlation":[.2]*4,"absolute_tau":[.2]*4,"strength_rank":[.7,.8,.9,.7],"realized_variation":[.01]*4,"variation_rank":[.7]*4,"completed_displacement":[.01]*4,"side":[1]*4,"eligible":[False,True,True,False]},columns=s.PANEL_COLS);onset,side=s.active(x);assert onset.tolist()==[False,True,False,False];assert side.tolist()==[1]*4;assert s.active(x,"direction_flip")[1].tolist()==[-1]*4;assert s.PREREG_SHA=="44e43bbb1bfbbb26f1983ce9557867fb7cac1819ebf9833a5e4d7ab299041a4f"
