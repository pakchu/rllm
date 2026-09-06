import numpy as np,pandas as pd,pytest
from training import build_high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay_support as s
def test_causal_rank_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(181,dtype=float));r=s.causal(x);assert r.iloc[:180].isna().all() and r.iloc[180]==1.
def test_consensus_requires_four_qualified_alts_and_rejects_tie():
 side,breadth,strength=s.consensus([1,1,1,1,-1,-1],[.8,.9,.9,.8,.9,.9],[1]*6,.75);assert (side,breadth)==(1,4) and strength==pytest.approx(.85);assert s.consensus([1,1,1,-1,-1,-1],[.9]*6,[1]*6,.75)[0]==0
def test_schema_is_outcome_blind_and_controls_frozen():
 assert s.PREREG_SHA=='98d0b3ae230334542aadb217cf53411c0291f0f78149d83616d1af74646ecfe4';assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_premium_sponsorship_gate','no_persistence_tail','no_variation_gate','negative_autocorrelation_consensus','one_block_stale_persistence','direction_flip','forced_long')
def test_primary_requires_premium_side_concordance():
 panel=pd.DataFrame({'source_valid':[True]*3,'consensus_side':[1]*3,'consensus_breadth':[4]*3,'consensus_strength':[.8]*3,'broad_side':[1]*3,'broad_breadth':[4]*3,'negative_side':[0]*3,'negative_breadth':[0]*3,'negative_strength':[np.nan]*3,'variation_rank':[.8]*3,'premium_displacement':[-.1,.1,-.1]})
 selected,side,_=s.active(panel);assert selected.tolist()==[False,True,False] and side[selected].tolist()==[1]
