import numpy as np, pandas as pd
from training import build_pposm_pre2024_oi_action_source as build

def _oi(end, periods=288):
 dates=pd.date_range(pd.Timestamp(end,tz='UTC')-pd.Timedelta(minutes=periods*5),periods=periods,freq='5min')
 return pd.DataFrame({'date':dates,'sum_open_interest':np.linspace(1000,1200,periods)})
def test_oi_feature_row_strict_prior_grid():
 r=build.feature_row(pd.Timestamp('2023-01-01T00:00:00Z'),_oi('2023-01-01T00:00:00Z'))
 assert r['source_valid'] is True
 assert r['oi60m_last_over_first_log']>0
 assert r['oi1440m_abs_logdiff_sum']>0
def test_oi_feature_row_rejects_nonpositive():
 f=_oi('2023-01-01T00:00:00Z'); f.loc[f.index[-1],'sum_open_interest']=0
 r=build.feature_row(pd.Timestamp('2023-01-01T00:00:00Z'),f)
 assert r['source_valid'] is False
 assert 'finite_positive' in r['invalid_reason']
