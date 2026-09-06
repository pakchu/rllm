import numpy as np,pandas as pd
from training import build_cross_asset_ticket_leadership_relay_support as support
def test_causal_normalization_and_rank_exclude_current():
 v=pd.Series(np.arange(1.,62.));n=support.causal_log_median_ratio(v);assert np.isfinite(n.iloc[60]);r=support.rank(pd.Series(np.arange(61.)));assert r.iloc[60]==1.
def test_primary_requires_positive_relative_leadership():
 f=pd.DataFrame({'ticket_leadership':[1.,-1.],'leadership_rank':[.8,.9],'btc_normalized_ticket':[1.,1.],'btc_ticket_rank':[.8,.8],'alt_median_normalized_ticket':[0.,0.],'alt_suppression_rank':[.8,.8],'source_valid':[True,True],'btc_return':[.01,-.01],'return_rank':[.7,.7],'volatility_rank':[.8,.8]});a,s=support.conditions(f,'primary');assert a.tolist()==[True,False];assert s[a].tolist()==[1.]
def test_bindings_and_outcomes_sealed():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;src=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in src;assert '"gross9_rows_opened":False' in src
