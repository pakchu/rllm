import numpy as np,pandas as pd
from training import build_fee_surface_persistence_relay_support as support
def test_broad_sign_requires_frozen_breadth():
 assert support.broad(np.array([1.,2.,3.,4.,-1.]),4)==1.;assert np.isnan(support.broad(np.array([1.,2.,3.,-1.,-2.]),4));assert support.broad(np.array([-1.,-2.,-3.,-4.,1.]),4)==-1.
def test_rank_is_strict_prior():
 r=support.rank(pd.Series(np.arange(91.)));assert r.iloc[90]==1.
def test_bindings_and_outcomes_sealed():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;src=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in src;assert '"gross9_rows_opened":False' in src
