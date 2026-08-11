import numpy as np
import pandas as pd
from training import build_high_volatility_negative_volume_index_signal_crossover_relay_support as s

def test_volume_indexes_update_only_selected_volume_direction():
 c=pd.Series([100.,110.,99.,108.9]);v=pd.Series([10.,9.,11.,10.]);valid=pd.Series([True]*4);x=s.volume_indexes(c,v,valid)
 assert x.nvi.tolist()==[1000.,1100.,1100.,1210.] and x.pvi.tolist()==[1000.,1000.,900.,900.]

def test_gap_resets_index():
 c=pd.Series([100.,110.,99.,100.]);v=pd.Series([10.,9.,11.,10.]);valid=pd.Series([True,True,False,True]);x=s.volume_indexes(c,v,valid);assert x.nvi.iloc[0]==1000 and x.nvi.iloc[1]==1100 and np.isnan(x.nvi.iloc[2]) and x.nvi.iloc[3]==1000

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():return pd.DataFrame({"source_valid":[True]*6,"nvi_difference":[-1.,-1.,1.,2.,-1.,-2.],"pvi_difference":[-1.,1.,1.,-1.,-1.,1.],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 p,side,_=s.active(panel(),"positive_volume_index");assert p.iloc[1] and side.iloc[1]==1 and p.iloc[3] and side.iloc[3]==-1
 stale,side,_=s.active(panel(),"one_bar_stale_crossover");assert stale.iloc[3] and side.iloc[3]==1
