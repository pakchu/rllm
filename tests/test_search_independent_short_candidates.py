import numpy as np
from training import search_independent_short_candidates as s


def data():
    return {'open':np.ones(12)*100,'end':np.ones(12)*100,'high':np.ones(12)*100,'low':np.ones(12)*100,'funding':np.zeros(12),
            'date':np.arange('2024-01-01T00:00','2024-01-01T01:00',dtype='datetime64[5m]').astype('datetime64[ns]'),
            'end_date':np.arange('2024-01-01T00:05','2024-01-01T01:05',dtype='datetime64[5m]').astype('datetime64[ns]')}


def test_stop_first_on_ambiguous_bar():
    d=data();d['high'][0]=102;d['low'][0]=98
    t=s.potential_trades(d,np.array([0]),1,.01,.01)
    assert t['barrier'][0] and t['exit'][0]==0 and t['exit_price'][0]==101
    assert np.isclose(t['gross_factor'][0],.99)


def test_no_parent_required_and_fee_accounting_matches_ledger():
    d=data();d['low'][1]=98.;d['end'][1]=99.
    t=s.potential_trades(d,np.array([0]),1,.01,.02)
    coarse=s.proxy(t,1,.0006);fine=s.exact(d,t,.0006)
    assert fine['trades']==1
    assert np.isclose(coarse['return_pct'],fine['return_pct'])


def test_forced_window_exit_and_nonoverlap():
    d=data();t=s.potential_trades(d,np.array([0,1,10]),3,None,None)
    assert t['exit'].tolist()==[12,12,12]
    assert len(s.schedule(t,np.ones(3,bool))['entry'])==1
    assert len(s.specs())==384


def test_positive_funding_credits_short_and_stops_at_exit():
    d=data();d['funding'][0]=1.
    t=s.potential_trades(d,np.array([0]),1,None,None)
    assert np.isclose(t['gross_factor'][0],1.01)
    assert np.isclose(s.exact(d,t,0)['return_pct'],1.)
    d['low'][0]=98.;d['funding'][1]=100.
    t=s.potential_trades(d,np.array([0]),1,.01,.02)
    assert t['exit'][0]==0
    assert np.isclose(t['gross_factor'][0],1.02)
