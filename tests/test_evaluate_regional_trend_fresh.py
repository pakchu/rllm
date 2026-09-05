import numpy as np,pandas as pd
from training import evaluate_regional_trend_fresh as s

def test_formula_requires_regional_confirmation():
 x=pd.DataFrame(index=pd.date_range('2026-06-01',periods=48,freq='1h'));x['vol24']=.01;x['mom168']=1.;x['kimchi_premium_change24']=.1
 p,raw=s.position(x);assert (raw>0).all();assert np.max(abs(p))<=1
 x['kimchi_premium_change24']=-.1;p,raw=s.position(x);assert not raw.any()
