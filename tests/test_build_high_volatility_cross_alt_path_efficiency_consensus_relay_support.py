import numpy as np,pandas as pd,pytest
from training import build_high_volatility_cross_alt_path_efficiency_consensus_relay_support as s
def test_causal_rank_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(181,dtype=float));r=s.causal(x);assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_consensus_requires_four_efficient_alts_and_rejects_tie():
 side,breadth,strength=s.consensus([1,1,1,1,-1,-1],[.8,.9,.9,.8,.9,.9],.75);assert (side,breadth)==(1,4) and strength==pytest.approx(.85);assert s.consensus([1,1,1,-1,-1,-1],[.9]*6,.75)[0]==0
def test_schema_is_outcome_blind_and_controls_frozen():
 assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_efficiency_tail','no_variation_gate','endpoint_return_magnitude_tail','one_block_stale_efficiency','direction_flip','forced_long')
