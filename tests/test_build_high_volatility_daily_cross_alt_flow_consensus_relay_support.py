import math,numpy as np
from training import build_high_volatility_daily_cross_alt_flow_consensus_relay_support as s
def test_consensus_geometry_is_frozen():
 assert s.consensus_geometry(np.array([.1,.2,.3,.4,-.1,-.2]))==(1,4,.25);side,breadth,strength=s.consensus_geometry(np.array([.1,.2,.3,-.4,-.1,-.2]));assert side==0 and breadth==3 and math.isnan(strength)
def test_schema_is_blind_unique_and_controls_frozen():
 assert len(s.CLOCK_COLUMNS)==len(set(s.CLOCK_COLUMNS));assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_strength_tail','no_variation_gate','three_of_six_consensus','one_day_stale_flow','direction_flip','forced_long')
