import numpy as np,pandas as pd
from training import build_cboe_direct_term_slope_rotation_relay_support as s
def test_rank_excludes_current():
 r=s.rank(pd.Series(range(127),dtype=float),252,126);assert np.isnan(r.iloc[125]);assert r.iloc[126]==1.
def test_direct_slope_and_rotation():
 f=pd.DataFrame({'observation_date':['2024-01-02','2024-01-03'],'SKEW_close':[100,100],'VVIX_close':[100,100],'VIX9D_close':[10,20],'VIX_close':[15,15],'VIX3M_close':[20,20]})
 old=s.pd.read_csv;s.pd.read_csv=lambda *a,**k:f.copy()
 oldsha=s.sha;s.sha=lambda p: s.prereg.SURFACE_SHA if p==s.prereg.SURFACE else s.prereg.MANIFEST_SHA
 try:x=s.load_surface()
 finally:s.pd.read_csv=old;s.sha=oldsha
 assert x.direct_slope.iloc[0]==np.log(.5);assert x.slope_rotation.iloc[1]==np.log(2)
def test_state_side_and_contract():
 f=pd.DataFrame({'source_valid':[True]*4,'direct_slope':[0]*4,'slope_rotation':[.1,.1,-.1,-.1],'rotation_rank':[.6,.7,.8,.6],'btc_variation':[1]*4,'btc_variation_rank':[.8]*4})
 a,side,_=s.states(f,'primary');assert a.tolist()==[False,True,False,False];assert side.tolist()==[-1,-1,1,1]
 assert s.PREREG_SHA=='7844b2295c0bb104455f9bba1c44392625f705ae4920a4fa0406387d01b56add'
