import pandas as pd
from training import build_high_volatility_trend_resolution_relay_support as s

def test_rank_excludes_current():
 ranks=s.strict_prior_midrank(pd.Series(range(61),dtype=float),lookback=90,minimum=60)
 assert ranks.iloc[:60].isna().all() and ranks.iloc[60]==1.0

def test_primary_requires_disagreement_to_agreement_under_high_variation():
 frame=pd.DataFrame({'source_valid':[True]*3,'return_5d':[.1,.2,.2],'return_20d':[-.2,.3,.3],'daily_realized_variation':[1.]*3,'variation_rank':[.5,.7,.8]})
 active,side,_=s.conditions(frame)
 assert active.tolist()==[False,True,False] and side.tolist()==[1,1,1]

def test_no_variation_gate_keeps_resolution_transition():
 frame=pd.DataFrame({'source_valid':[True,True],'return_5d':[.1,.2],'return_20d':[-.2,.3],'daily_realized_variation':[1.,1.],'variation_rank':[.5,.4]})
 assert s.conditions(frame,'no_variation_gate')[0].tolist()==[False,True]
