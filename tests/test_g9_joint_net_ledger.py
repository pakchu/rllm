import numpy as np
from training import g9_joint_net_ledger as l


def data():
    return {'open':np.array([100.,100.]),'end':np.array([100.,110.]),'high':np.array([105.,120.]),'low':np.array([95.,90.]),'funding':np.zeros(2),
            'date':np.array(['2026-01-01','2026-01-02'],dtype='datetime64[ns]'),
            'end_date':np.array(['2026-01-02','2027-01-01'],dtype='datetime64[ns]')}


def test_opposite_units_offset_before_costs():
    rows,_=l.simulate(data(),np.array([[1.,-1.]]*2),np.array([[1,1],[0,0]],bool),np.full((2,2),np.nan),np.array([[1.,1.]]),['a','b'],cost=.001)
    assert rows[0]['return_pct']==0 and rows[0]['fees_pct_initial']==0


def test_barrier_price_used_not_next_open():
    rows,_=l.simulate(data(),np.array([[1.],[0.]]),np.array([[1],[0]],bool),np.array([[105.],[np.nan]]),np.array([[1.]]),['a'],cost=0)
    assert np.isclose(rows[0]['return_pct'],5)
    assert rows[0]['orders']==2


def test_carry_units_not_resized_by_other_sleeve():
    rows,_=l.simulate(data(),np.array([[1.,0.],[1.,0.]]),np.array([[1,1],[0,1]],bool),np.full((2,2),np.nan),np.array([[1.,1.]]),['a','b'],cost=0)
    assert np.isclose(rows[0]['return_pct'],10)
    assert rows[0]['orders']==2


def test_dead_portfolio_never_reopens():
    d=data();d['end']=np.array([1.,110.]);d['open']=np.array([100.,1.])
    rows,paths=l.simulate(d,np.ones((2,1)),np.ones((2,1),bool),np.full((2,1),np.nan),np.array([[4.5]]),['a'],cost=0)
    assert rows[0]['insolvent'] and np.all(paths==0)


def test_barrier_breaking_hedge_exposes_remaining_side_in_mdd():
    d=data();d['high'][0]=150.;d['low'][0]=50.
    rows,_=l.simulate(d,np.array([[1.,-1.],[0.,0.]]),np.array([[1,1],[0,0]],bool),np.array([[105.,np.nan],[np.nan,np.nan]]),np.array([[1.,1.]]),['a','b'],cost=0)
    assert rows[0]['mdd_pct']>40


def test_intrabar_ruin_is_absorbing_even_if_close_recovers():
    d=data();d['low'][0]=0
    rows,paths=l.simulate(d,np.ones((2,1)),np.ones((2,1),bool),np.full((2,1),np.nan),np.array([[4.5]]),['a'],cost=0)
    assert rows[0]['insolvent'] and rows[0]['mdd_pct']==100
    assert np.all(paths==0)


def test_active_event_cap_is_post_fee():
    d=data();d['high']=np.array([100.,110.]);d['low']=np.array([100.,100.])
    rows,_=l.simulate(d,np.ones((2,1)),np.ones((2,1),bool),np.full((2,1),np.nan),np.array([[4.5]]),['a'],cost=.001)
    assert rows[0]['max_open_net_exposure']<=4.5+1e-10
    assert rows[0]['cap_interventions']>0
