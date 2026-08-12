import numpy as np
from training import build_high_volatility_alt_leadership_rotation_relay_support as s
def test_unique_leader_geometry_is_deterministic():
 i,v,c=s.leader_geometry(np.array([1,2,3,4,5,-10.]));assert (i,v)==(5,-10.);assert 1/6<=c<=1;assert s.leader_geometry(np.array([1,2,3,4,10,-10.]))[0]==-1
def test_schema_and_controls_are_frozen_and_blind():
 assert len(s.CLOCK_COLUMNS)==len(set(s.CLOCK_COLUMNS));assert not {"pnl","funding","execution_price","gross9"}.intersection(x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS));assert s.CONTROLS==("no_variation_gate","identity_change_without_directional_handoff","current_leader_persistence","one_block_stale_geometry","direction_flip","forced_long")
