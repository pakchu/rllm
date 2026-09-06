import numpy as np
import pandas as pd
from training import build_high_volatility_spot_flow_cross_quantilogram_relay_support as s


def test_cross_quantilogram_detects_tail_predictability():
    predictor=np.tile(np.arange(479)%4,1).astype(float)
    response=np.roll(predictor,0)
    lower,upper=s.cross_quantilogram(predictor,response)
    assert lower>.9 and upper>.9


def test_rank_excludes_current():
    ranks=s.strict_prior_midrank(pd.Series(range(181),dtype=float))
    assert ranks.iloc[:180].isna().all() and ranks.iloc[180]==1.


def test_primary_onset_uses_active_tail_and_confirmation():
    frame=pd.DataFrame({'source_valid':[True]*4,'direction_confirmed':[True]*4,'active_score':[.1]*4,'active_score_rank':[.7,.8,.9,.7],'variation_rank':[.7]*4,'final_hour_spot_flow':[.1,.1,-.1,-.1],'same_minute_lower':[.1]*4,'same_minute_upper':[.1]*4,'same_minute_lower_rank':[.8]*4,'same_minute_upper_rank':[.8]*4})
    onset,side=s.conditions(frame,'primary')
    assert onset.tolist()==[False,True,False,False]
    assert side.tolist()==[1,1,-1,-1]


def test_contract_is_frozen():
    assert s.POLICY_ID=='HVSFCQ-8'
    assert s.PREREG_SHA=='dac611d36a71b6536c0d5d96a4e18c2bcc1106f6ee29db24ea252e931baee76f'
    assert s.CONTROLS==('no_cross_quantilogram_tail','no_variation_gate','same_minute_cross_quantilogram','one_decision_stale_dependence','direction_flip','forced_long')
