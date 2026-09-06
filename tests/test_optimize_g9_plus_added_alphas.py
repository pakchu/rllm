import numpy as np
import pandas as pd
from training import optimize_g9_plus_added_alphas as o


def test_weights_controls_and_grid_deterministic():
    w,labels=o.allocation_grid();w2,l2=o.allocation_grid()
    assert np.array_equal(w,w2) and labels==l2
    assert w[0].sum()==4. and w[1].sum()==4.5
    assert len(labels)==len(set(labels))
    assert (w>=0).all()


def test_barrier_is_held_during_exit_bar_and_then_cleared():
    dates=pd.date_range('2026-06-01',periods=4,freq='5min')
    r={'sleeves':{name:{'trades':[]} for name in o.G9}}
    r['sleeves'][o.G9[0]]['trades']=[{'entry_date':str(dates[0]),'exit_date':str(dates[1]),'side':'LONG','exit_kind':'barrier','exit_price':105}]
    p,e,b=o.clock_arrays(r,dates)
    assert p[:,0].tolist()==[1,1,0,0]
    assert e[:,0].tolist()==[True,False,True,False]
    assert b[1,0]==105
