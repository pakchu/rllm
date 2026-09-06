import numpy as np,pandas as pd
from training import build_high_volatility_cross_alt_illiquidity_impulse_consensus_relay_support as s

def test_causal_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(181,dtype=float));m=s.causal(x,'median');assert m.iloc[:180].isna().all();assert m.iloc[180]==89.5

def test_consensus_requires_four_tail_alts_and_rejects_tie():
 assert s.consensus([1,1,1,1,-1,-1],[.9,.9,.9,.9,.9,.9],.8)==(1,4,.9)
 assert s.consensus([1,1,1,-1,-1,-1],[.9]*6,.8)[0]==0

def test_schema_is_outcome_blind_and_controls_frozen():
 assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS))
 assert s.CONTROLS==('no_impulse_tail','no_variation_gate','raw_return_tail_consensus','one_block_stale_impulse','direction_flip','forced_long')
