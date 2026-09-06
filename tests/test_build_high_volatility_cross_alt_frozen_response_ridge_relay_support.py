import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_frozen_response_ridge_relay_support as s
def test_causal_rank_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(181,dtype=float));r=s.causal(x);assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_ridge_intercept_is_unpenalized_and_deterministic():
 x=pd.DataFrame({'a':np.arange(10,dtype=float),'b':np.arange(10,dtype=float)**2});y=1+2*x.a-.1*x.b;m=s.fit_ridge(x,('a','b'),y);p=s.predict(x,m);assert np.isfinite(p).all() and abs(p.mean()-y.mean())<1e-10
def test_schema_is_oos_outcome_blind_and_controls_frozen():
 assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_prediction_tail','no_variation_gate','alt_returns_only_ridge','one_block_stale_prediction','direction_flip','forced_long')
