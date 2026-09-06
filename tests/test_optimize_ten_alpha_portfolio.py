import numpy as np
import pandas as pd
from training import optimize_ten_alpha_portfolio as o


def test_grid_includes_both_shorts_and_prior_controls():
    w,n=o.grid();w2,n2=o.grid()
    assert np.array_equal(w,w2) and n==n2
    assert w.shape[1]==10 and len(w)>500
    assert np.all(w<=o.LIMITS+1e-12)
    assert 'g9_macro1_d0.5_r0.5' in n
    assert np.array_equal(w[0],o.G9) and np.array_equal(w[1],o.PARENT)


def test_recent_short_barrier_mapping():
    dates=pd.date_range('2026-06-01',periods=3,freq='5min').to_numpy()
    d={'date':dates,'end_date':dates+np.timedelta64(5,'m'),'low':np.ones(3)*90,'high':np.ones(3)*110}
    p,e,b=o.add_short_clock(d,[{'entry_date':str(dates[0]),'exit_date':str(dates[1]),'exit_price':98.,'barrier':True}])
    assert p.tolist()==[-1,-1,0]
    assert e.tolist()==[True,False,True]
    assert b[1]==98.
