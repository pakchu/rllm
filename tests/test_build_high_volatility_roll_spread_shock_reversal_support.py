import numpy as np
import pandas as pd
from training import build_high_volatility_roll_spread_shock_reversal_support as support

def test_roll_statistics_recovers_negative_covariance_spread():
    returns=np.tile([.01,-.01],48)[:95];closes=100*np.exp(np.r_[0,np.cumsum(returns)])
    covariance,spread,variation,completed=support.roll_statistics(closes,100.,np.zeros(480))
    assert covariance<0 and spread>0 and variation>0 and np.isfinite(completed)

def test_onset_uses_previous_source_valid_block():
    eligible=pd.Series([False,True,False,True]);valid=pd.Series([True,True,False,True])
    assert support.previous_valid_onset(eligible,valid).tolist()==[False,True,False,False]

def test_artifact_columns_are_outcome_blind():
    names={x.lower() for x in (*support.PANEL_COLUMNS,*support.CLOCK_COLUMNS)}
    assert not names.intersection({"pnl","funding","execution_price","gross9"})
