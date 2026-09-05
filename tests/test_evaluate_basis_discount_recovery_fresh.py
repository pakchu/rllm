import numpy as np,pandas as pd
from training import evaluate_basis_discount_recovery_fresh as s

def test_fixed_candidate_formula_and_cap():
 n=48;x=pd.DataFrame(index=pd.date_range('2026-06-01',periods=n,freq='1h'))
 x['vol24']=.01;x['premium_z168']=-2.;x['premium_change6']=.1;x['premium_change6_z168']=2.;x['flow6']=.03
 p,c=s.position(x);assert np.max(np.abs(p))<=1;assert np.count_nonzero(c['recovery'])==n
 assert s.DESIGN['historical_identity']=='mix_55_61_0.25'
