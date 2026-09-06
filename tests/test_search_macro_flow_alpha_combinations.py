import pandas as pd
import numpy as np
from training import search_macro_flow_alpha_combinations as s


def sample():
 dates=pd.date_range('2023-01-01',periods=3000,freq='5min');t=np.arange(len(dates))
 return pd.DataFrame({'date':dates,'dxy':100+t*.001,'usdkrw':1200+t*.01,'kimchi_premium':t*.00001,'dxy_available':1.,'usdkrw_available':1.,'kimchi_available':1.})


def test_macro_extra_hour_lag_blocks_current_period():
 m=sample();index=pd.date_range('2023-01-02',periods=100,freq='1h');decision=index[50]
 x=s.macro_features(m,index)
 altered=m.copy();altered.loc[altered.date>=decision-pd.Timedelta('1h'),'dxy']*=2
 y=s.macro_features(altered,index)
 pd.testing.assert_series_equal(x.loc[decision],y.loc[decision])


def test_missing_currency_is_not_zero_price():
 m=sample();m['dxy']=0.;m['dxy_available']=0.
 x=s.macro_features(m,pd.date_range('2023-01-01',periods=100,freq='1h'))
 assert x.dxy_change24.isna().all()
 assert not x.dxy_valid.any()
