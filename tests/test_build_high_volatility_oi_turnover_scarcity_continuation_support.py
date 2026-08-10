import pandas as pd
from training import build_high_volatility_oi_turnover_scarcity_continuation_support as s
def test_onset_uses_previous_valid():assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]
def test_schema_blind():assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({'pnl','funding','execution_price','gross9'})
def test_controls_match_prereg():assert s.CONTROLS==('no_scarcity_tail_gate','no_variation_gate','oi_level_only','turnover_scarcity_only','one_block_stale_geometry','direction_flip','forced_long')
