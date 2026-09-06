import numpy as np
from training import optimize_added_alpha_portfolio as m


def test_simplex_grid():
    w=m.grid()
    assert w.shape==(231,3)
    assert np.allclose(w.sum(axis=1),1)
    assert (w>=0).all()


def test_other_sleeve_updates_do_not_resize_fixed_units():
    d={'open':np.array([100.,110.,120.]),'end':np.array([110.,120.,130.]),
       'high':np.array([110.,120.,130.]),'low':np.array([100.,110.,120.]),
       'funding':np.zeros(3),'date':np.array(['2026-01-01','2026-01-02','2026-01-03'],dtype='datetime64[ns]'),
       'end_date':np.array(['2026-01-02','2026-01-03','2026-01-04'],dtype='datetime64[ns]')}
    targets=np.array([[0.,1.,0.]]*3)
    events=np.array([[1,1,1],[1,0,1],[1,0,1]],bool)
    result=m.simulate(d,targets,events,np.array([[.5,.5,0.]]),0)[0]
    assert np.isclose(result['return_pct'],15.)
    assert result['orders_including_liquidation']==2


def test_opposing_units_net_before_fees():
    d={'open':np.array([100.]),'end':np.array([101.]),'high':np.array([102.]),'low':np.array([99.]),
       'funding':np.array([1.]),'date':np.array(['2026-01-01'],dtype='datetime64[ns]'),
       'end_date':np.array(['2027-01-01'],dtype='datetime64[ns]')}
    r=m.simulate(d,np.array([[1.,-1.,0.]]),np.ones((1,3),bool),np.array([[.5,.5,0.]]),.001)[0]
    assert r['return_pct']==0 and r['fees_pct_initial']==0 and r['funding_pct_initial']==0
