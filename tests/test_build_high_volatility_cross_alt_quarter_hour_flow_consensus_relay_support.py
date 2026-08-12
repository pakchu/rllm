import math,numpy as np
import pandas as pd
from training import build_high_volatility_cross_alt_quarter_hour_flow_consensus_relay_support as s
def test_consensus_geometry_is_fixed():
 assert s.consensus_geometry(np.array([.1,.2,.3,.4,-.1,-.2]))==(1,4,.25);side,breadth,strength=s.consensus_geometry(np.array([.1,.2,.3,-.4,-.1,-.2]));assert side==0 and breadth==3 and math.isnan(strength)
def test_schema_is_blind_unique_and_controls_frozen():
 assert len(s.CLOCK_COLUMNS)==len(set(s.CLOCK_COLUMNS));assert not {"pnl","funding","execution_price","gross9"}.intersection(x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==("no_strength_tail","no_variation_gate","three_of_six_consensus","one_quarter_stale_geometry","direction_flip","forced_long")
def test_stale_geometry_shifts_all_frozen_geometry_columns():
 panel=pd.DataFrame({c:[0,1] for c in s.PANEL_COLUMNS})
 panel["source_valid"]=[False,True];panel["consensus_side"]=[-1,1]
 active,side,used=s.active(panel,"one_quarter_stale_geometry")
 assert not active.any() and pd.isna(used.loc[0,"source_valid"])
 assert used.loc[1,"consensus_side"]==-1 and side.loc[1]==-1
