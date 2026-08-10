import numpy as np,pandas as pd
from training import build_high_volatility_signed_variance_feedback_relay_support as s
def test_feedback_score_tracks_signed_shock_to_next_energy():
 returns=np.array(([.01,.03]+[.001,-.001]*47)[:95]);closes=100*np.exp(np.r_[0,np.cumsum(returns)]);score,magnitude,variation,contemporary=s.feedback_statistics(closes);assert score>0 and magnitude==abs(score) and variation>0 and np.isfinite(contemporary)
def test_onset_uses_previous_valid():
 assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]
def test_schema_is_outcome_blind():
 assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({'pnl','funding','execution_price','gross9'})
