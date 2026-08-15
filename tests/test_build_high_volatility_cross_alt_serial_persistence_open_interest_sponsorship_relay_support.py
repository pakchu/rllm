import numpy as np,pandas as pd,pytest
from training import build_high_volatility_cross_alt_serial_persistence_open_interest_sponsorship_relay_support as s
def test_causal_rank_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(181,dtype=float));r=s.causal(x);assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_consensus_requires_four_qualified_alts_and_rejects_tie():
 side,breadth,strength=s.consensus([1,1,1,1,-1,-1],[.8,.9,.9,.8,.9,.9],[1]*6,.75);assert (side,breadth)==(1,4) and strength==pytest.approx(.85);assert s.consensus([1,1,1,-1,-1,-1],[.9]*6,[1]*6,.75)[0]==0
def test_schema_is_outcome_blind_and_controls_frozen():
 assert s.PREREG_SHA=='ae55ad43aeee388bf80eb33963a9fac32a3f116ac906e8b1b73a51041b7f6e69';assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_open_interest_sponsorship_gate','no_persistence_tail','no_variation_gate','negative_autocorrelation_consensus','one_block_stale_persistence','direction_flip','forced_long')
def test_primary_requires_strict_oi_sponsorship():
 panel=pd.DataFrame({'source_valid':[True]*3,'consensus_side':[1]*3,'consensus_breadth':[4]*3,'consensus_strength':[.8]*3,'broad_side':[1]*3,'broad_breadth':[4]*3,'negative_side':[0]*3,'negative_breadth':[0]*3,'negative_strength':[np.nan]*3,'variation_rank':[.8]*3,'oi_change':[-.1,.1,-.1]})
 selected,side,_=s.active(panel);assert selected.tolist()==[False,True,False] and side[selected].tolist()==[1]
