import numpy as np,pandas as pd
from training import build_spot_perpetual_variance_transfer_asymmetry_support as support

def frame():
 return pd.DataFrame({'signal_valid':[True]*5,'relocation':[.3,.2,.3,.3,.3],'relocation_rank':[.8,.9,.6,.8,.8],'spot_second_share':[.7,.6,.7,.4,.7],'perp_second_share':[.4,.4,.4,.2,.4],'spot_final2_return':[.01,-.01,.01,.01,.01],'perp_final2_return':[.02,-.02,-.01,.02,.02],'btc_realized_variation_rank':[.7,.8,.9,.9,.4]})

def test_primary_follows_common_final_two_hour_direction():
 active,side=support.conditions(frame(),'primary');assert active.tolist()==[True,True,False,False,False];assert side[active].tolist()==[1.,-1.]

def test_controls_are_diagnostic_only():
 f=frame();assert support.CONTROLS==('no_volatility_gate','no_relocation_tail','no_opposite_half_geometry','spot_direction_only','one_block_stale_relocation','direction_flip');assert support.conditions(f,'no_relocation_tail')[0].tolist()==[True,True,False,False,False];assert support.conditions(f,'no_opposite_half_geometry')[0].tolist()==[True,True,False,True,False];assert support.conditions(f,'spot_direction_only')[0].tolist()==[True,True,False,False,False];active,side=support.conditions(f,'direction_flip');assert side[active].tolist()==[-1.,1.]

def test_causal_rank_excludes_current_value():
 v=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(v,90,60);assert r.iloc[60]==1.

def test_builder_binds_sources_and_seals_outcomes():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;assert support.sha(support.SPOT)==support.SPOT_SHA;assert support.sha(support.PERP)==support.PERP_SHA;s=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in s;assert '"gross9_rows_opened":False' in s
