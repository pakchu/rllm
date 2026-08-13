import numpy as np,pandas as pd,pytest
from training import build_high_volatility_horizontal_visibility_direction_relay_support as s
def test_visibility_direction_rises_and_falls():
 peaks=np.linspace(5.,10.,48);x=np.empty(96);x[0::2]=peaks;x[1::2]=0.;a,b,n=s.visibility_direction(x);assert a>0 and b>0 and n>=24;a,b,n=s.visibility_direction(x[::-1]);assert a<0 and b<0 and n>=24
def test_visibility_fails_closed():assert np.isnan(s.visibility_direction(np.ones(95))[0])
def test_rank_excludes_current():
 r=s.prior_rank(pd.Series(range(181),dtype=float));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def test_onset_side_controls_and_hash():
 d=pd.date_range("2024-07-01T06:00:00Z",periods=4,freq="8h");x=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"visibility_direction":[.5]*4,"unweighted_direction":[.4]*4,"nonadjacent_edges":[50]*4,"direction_strength_rank":[.7,.8,.9,.7],"unweighted_rank":[.8]*4,"realized_variation":[.01]*4,"variation_rank":[.7]*4,"block_return":[.01]*4,"side":[1]*4,"eligible":[False,True,True,False]},columns=s.PANEL_COLS);onset,side=s.active(x);assert onset.tolist()==[False,True,False,False];assert side.tolist()==[1]*4;assert s.active(x,"direction_flip")[1].tolist()==[-1]*4;assert s.PREREG_SHA=="7a19bd2f228f3075b24a319be3dce843723936d3ed3b07a245a346df126f52b7"
