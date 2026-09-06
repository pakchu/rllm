import numpy as np,pandas as pd
from training import build_high_volatility_basis_regime_transition_relay_support as s
def test_causal_stat_excludes_current_and_obeys_floor():
 x=pd.Series(np.arange(145,dtype=float));m=s.causal_stat(x,168,144,'median');assert m.iloc[:144].isna().all();assert m.iloc[144]==71.5
def test_schema_is_blind_unique_and_controls_frozen():
 assert len(s.CLOCK_COLUMNS)==len(set(s.CLOCK_COLUMNS));assert not {'pnl','funding','execution_price','gross9','future_return'}.intersection(c.lower() for c in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==('no_residual_tail','no_variation_gate','raw_basis_zero_cross','one_hour_stale_transition','direction_flip','forced_long')
