import numpy as np,pandas as pd
from training import build_high_volatility_funding_acceleration_divergence_relay_support as s
def test_causal_rank_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(181,dtype=float));r=s.causal(x);assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_onset_requires_prior_valid_ineligible_cycle():
 state=pd.Series([False,True,True,False,True]);valid=pd.Series([True]*5);assert s.onset(state,valid).tolist()==[False,True,False,False,True]
def test_schema_is_outcome_blind_and_controls_frozen():
 assert not {'pnl','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_change_tail','no_variation_gate','funding_change_direction','one_settlement_stale_change','direction_flip','forced_long')
