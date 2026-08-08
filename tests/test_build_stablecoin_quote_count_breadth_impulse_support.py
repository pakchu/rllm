import numpy as np,pandas as pd
from training import build_stablecoin_quote_count_breadth_impulse_support as support

def frame():
 return pd.DataFrame({'signal_valid':[True]*5,'breadth_impulse':[.1,.1,.1,-.1,.1],'entropy_impulse':[.1,.1,-.1,.1,.1],'impulse_rank':[.8,.9,.8,.9,.6],'signed_flow_BTCUSDC':[1.,-1.,1.,1.,1.],'signed_flow_BTCFDUSD':[.5,-.5,-.5,.5,.5],'btc_realized_variation_rank':[.7,.8,.9,.9,.9]})

def test_primary_follows_alternative_flow_on_breadth_expansion():
 active,side=support.conditions(frame(),'primary');assert active.tolist()==[True,True,False,False,False];assert side[active].tolist()==[1.,-1.]

def test_controls_are_diagnostic_only():
 f=frame();assert support.CONTROLS==('no_volatility_gate','no_impulse_tail','no_entropy_confirmation','one_block_stale_breadth','direction_flip');assert support.conditions(f,'no_impulse_tail')[0].tolist()==[True,True,False,False,True];assert support.conditions(f,'no_entropy_confirmation')[0].tolist()==[True,True,False,False,False];active,side=support.conditions(f,'direction_flip');assert side[active].tolist()==[-1.,1.]

def test_causal_rank_excludes_current_value():
 v=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(v,90,60);assert r.iloc[60]==1.

def test_builder_binds_sources_and_seals_outcomes():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;assert support.sha(support.PANEL)==support.PANEL_SHA;assert support.sha(support.PRICE)==support.PRICE_SHA;s=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in s;assert '"gross9_rows_opened":False' in s
