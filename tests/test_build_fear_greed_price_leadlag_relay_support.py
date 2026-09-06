import numpy as np,pandas as pd
from training import build_fear_greed_price_leadlag_relay_support as support

def frame():
 return pd.DataFrame({'signal_valid':[True]*5,'sentiment_change':[5.,-5.,5.,5.,5.],'sentiment_change_rank':[.8,.9,.5,.8,.8],'btc_day_return':[-.01,.01,.01,-.01,-.01],'btc_variation_rank':[.7,.8,.9,.4,.9]})

def test_primary_follows_price_when_sentiment_disagrees():
 active,side=support.conditions(frame(),'primary');assert active.tolist()==[True,True,False,False,True];assert side[active].tolist()==[-1.,1.,-1.]

def test_controls_are_diagnostic_only():
 f=frame();assert support.CONTROLS==('no_volatility_gate','no_sentiment_change_tail','no_direction_disagreement','sentiment_direction','one_day_stale_sentiment_change','direction_flip');assert support.conditions(f,'no_volatility_gate')[0].tolist()==[True,True,False,True,True];assert support.conditions(f,'no_sentiment_change_tail')[0].tolist()==[True,True,False,False,True];assert support.conditions(f,'no_direction_disagreement')[0].tolist()==[True,True,False,False,True];active,side=support.conditions(f,'sentiment_direction');assert side[active].tolist()==[1.,-1.,1.];active,side=support.conditions(f,'direction_flip');assert side[active].tolist()==[1.,-1.,1.]

def test_causal_rank_excludes_current_value():
 v=pd.Series(np.arange(61,dtype=float));r=support.strict_prior_midrank(v,90,60);assert r.iloc[60]==1.

def test_builder_binds_sources_and_seals_outcomes():
 assert support.sha(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;assert support.sha(support.SENTIMENT)==support.SENTIMENT_SHA;assert support.sha(support.PRICE)==support.PRICE_SHA;s=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened":False' in s;assert '"gross9_rows_opened":False' in s
