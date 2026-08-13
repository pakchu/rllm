import numpy as np,pandas as pd
from training import build_high_volatility_alt_mst_hub_ignition_relay_support as s
def test_mst_star_has_unique_hub_and_unit_centralization():
 c=np.eye(6);c[0,1:]=c[1:,0]=.9
 for i in range(1,6):
  for j in range(i+1,6):c[i,j]=c[j,i]=0.
 hub,degree,central=s.mst_topology(c);assert hub==s.ALTS[0] and degree==5 and central==1.
def test_rank_excludes_current():
 r=s.prior_rank(pd.Series(range(181),dtype=float));assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_onset_controls_and_hash():
 d=pd.date_range("2024-07-01T05:00:00Z",periods=4,freq="8h");x=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"hub_symbol":[s.ALTS[0]]*4,"hub_degree":[5]*4,"degree_centralization":[1.]*4,"centralization_rank":[.7,.8,.9,.7],"hub_final_hour_return":[.01]*4,"equal_weight_final_hour_return":[-.01]*4,"btc_realized_variation":[.01]*4,"variation_rank":[.7]*4,"side":[1]*4,"equal_weight_side":[-1]*4,"eligible":[False,True,True,False]},columns=s.PANEL_COLS);onset,side,_=s.active(x);assert onset.tolist()==[False,True,False,False] and side.tolist()==[1]*4;assert s.active(x,"equal_weight_alt_direction")[1].tolist()==[-1]*4;assert s.PREREG_SHA=="a41fdf5ce071ed241172a2d2b1247d952d8992a55c7dcf64d3ff2b12127e1654"
