import numpy as np
import pandas as pd
from training import search_regime_diverse_alpha_combinations as study


def test_annual_fit_purges_future_maturity():
    dates=pd.DatetimeIndex(['2020-01-01','2020-12-30','2020-12-31','2021-01-01','2022-01-01'])
    train,test=study.annual_masks(dates,24,2021)
    assert train.tolist()==[False,True,False,False,False]
    assert test.tolist()==[False,False,False,True,False]
    assert all(dates[train]+pd.Timedelta(hours=24,minutes=5)<pd.Timestamp('2021-01-01'))


def test_trailing_three_year_window_is_exact():
    dates=pd.date_range('2020-01-01', '2026-01-01',freq='1D')
    train,test=study.annual_masks(dates,24,2025)
    assert dates[train].min()==pd.Timestamp('2022-01-01')
    assert dates[test].min()==pd.Timestamp('2025-01-01')
    assert dates[test].max()<pd.Timestamp('2026-01-01')


def test_declared_mixtures_preserve_net_cap():
    a=np.array([1,-1,1,0]);b=np.array([-1,1,0,-1])
    for w in [.25,.5,.75]:
        net=w*a+(1-w)*b
        assert np.abs(net).max()<=1
        assert net[0]==2*w-1
