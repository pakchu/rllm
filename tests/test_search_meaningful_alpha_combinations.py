import numpy as np
import pandas as pd
import pytest
from training import search_meaningful_alpha_combinations as s


def blocks(n=4):
    op=np.full(n,100.)
    return {'date':pd.date_range('2023-01-01 00:05',periods=n,freq='1h').to_numpy(),
            'open':op,'end':op.copy(),'high':op.copy(),'low':op.copy(),
            'funding':np.zeros(n),'debit':np.zeros(n),'credit':np.zeros(n),
            'op5':np.full((n,12),100.),'cl5':np.full((n,12),100.),
            'hi5':np.full((n,12),100.),'lo5':np.full((n,12),100.),'fund5':np.zeros((n,12))}


def test_cash_and_offset_are_exactly_flat():
    d=blocks();net=.5*np.ones(4)+.5*(-np.ones(4))
    r=s.simulate(d,net,fine=True)
    assert r['equity'][0]==1
    assert r['fees_pct_initial'][0]==0
    assert r['entry_episodes'][0]==0


def test_fees_entry_exit_and_sign_reversal():
    d=blocks(1);r=s.simulate(d,np.ones(1),cost=.001,fine=True)
    assert r['equity'][0]==pytest.approx(.998)
    d=blocks(2);r=s.simulate(d,[1,-1],cost=.001)
    assert r['entry_episodes'][0]==2
    assert r['fees_pct_initial'][0]>0.399
    assert r['equity'][0]<.997


def test_funding_side_and_intrabar_risk():
    d=blocks(1);d['funding'][0]=1.;d['debit'][0]=1.;d['fund5'][0,0]=1.
    a=s.simulate(d,[1],cost=0,fine=True);b=s.simulate(d,[-1],cost=0,fine=True)
    assert a['equity'][0]==pytest.approx(.99)
    assert b['equity'][0]==pytest.approx(1.01)
    d=blocks(1);d['hi5'][0,0]=110;d['lo5'][0,0]=80
    r=s.simulate(d,[1],cost=0,fine=True)
    assert r['mdd_pct'][0]==pytest.approx((1-.8/1.1)*100)


def test_screen_and_fine_have_identical_terminal_equity():
    d=blocks(3);d['open'][:]=[100,101,102];d['end'][:]=[101,102,103]
    for i in range(3):
        d['op5'][i,:]=d['open'][i]
        d['hi5'][i,:]=d['high'][i]=105
        d['lo5'][i,:]=d['low'][i]=95
    p=np.array([[1,.5],[.2,-.5],[0,1]])
    a=s.simulate(d,p);b=s.simulate(d,p,fine=True)
    np.testing.assert_allclose(a['equity'],b['equity'])
    np.testing.assert_allclose(a['fees_pct_initial'],b['fees_pct_initial'])


def test_decision_never_uses_current_hour_bars():
    dates=pd.date_range('2020-01-01',periods=9000,freq='5min')
    c=100+np.arange(len(dates))*.001
    m=pd.DataFrame({'date':dates,'open':c,'high':c+.1,'low':c-.1,'close':c,'volume':1.,'quote_asset_volume':100.,'taker_buy_quote':50.,'number_of_trades':10.})
    f=pd.DataFrame({'date':dates[::96],'funding_rate':.0001})
    decision=pd.Timestamp('2020-02-01')
    a=s.features(m,f)
    altered=m.copy();altered.loc[altered.date>=decision,'close']*=2
    b=s.features(altered,f)
    pd.testing.assert_series_equal(a.loc[decision],b.loc[decision])


def test_net_risk_limit_and_rebalance_clock():
    with pytest.raises(ValueError):s.simulate(blocks(),[2]*4)
    d=pd.date_range('2023-01-01',periods=10,freq='1h')
    p=s.hold_signal(np.arange(10),6,d)
    assert p.tolist()==[0]*6+[6]*4


def test_boundary_block_cannot_leak_into_selection():
    d={'date':np.array(['2023-12-31T22:05','2023-12-31T23:05'],dtype='datetime64[m]'),
       'end_date':np.array(['2023-12-31T23:05','2024-01-01T00:05'],dtype='datetime64[m]')}
    assert s.window_mask(d,'2023-01-01','2024-01-01').tolist()==[True,False]


def test_missing_funding_event_fails_closed():
    d=pd.date_range('2020-01-01',periods=6,freq='8h')
    f=pd.DataFrame({'date':d,'funding_rate':.0001})
    s.validate_funding(f,d[-1])
    with pytest.raises(ValueError,match='gap'):
        s.validate_funding(f.drop(index=2),d[-1])


def test_bankrupt_short_stays_dead_after_reversal():
    d=blocks(2);d['open'][:]=[100,250];d['end'][:]=[250,100]
    d['high'][:]=[250,250];d['low'][:]=[100,100]
    r=s.simulate(d,[-1,-1],cost=0)
    assert r['equity'][0]==0
