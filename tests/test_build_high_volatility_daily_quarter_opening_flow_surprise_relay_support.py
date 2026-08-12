import math,numpy as np,pandas as pd
from training import build_high_volatility_daily_quarter_opening_flow_surprise_relay_support as s
def test_geometry_and_causal_baseline_are_frozen():
 assert s.geometry(np.array([.1,.2,.3,.4,-.1,-.2]))==(1,4,.25);x=pd.Series(np.arange(25,dtype=float));m=s.causal(x,30,24,'median');assert m.iloc[:24].isna().all() and m.iloc[24]==11.5
def test_schema_is_blind_unique_and_controls_frozen():
 assert len(s.CLOCK_COLUMNS)==len(set(s.CLOCK_COLUMNS));assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_strength_tail','no_variation_gate','raw_opening_flow_level','one_day_stale_surprise','direction_flip','forced_long')
