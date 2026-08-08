import numpy as np
import pandas as pd
from training import build_usdjpy_carry_volatility_relay_support as support


def frame():
    return pd.DataFrame({"signal_valid":[True]*4,"fx_return":[.01,-.02,.03,-.04],"fx_return_z":[1.2,-1.3,.5,-1.5],"btc_realized_variation_rank":[.7,.8,.9,.4]})


def test_primary_direction_and_frozen_gates():
    active,side=support.conditions(frame(),"primary")
    assert active.tolist()==[True,True,False,False]
    assert side[active].tolist()==[1.0,-1.0]


def test_controls_are_diagnostic_transformations():
    f=frame();assert support.conditions(f,"no_volatility_gate")[0].tolist()==[True,True,False,True]
    assert support.conditions(f,"no_fx_tail")[0].tolist()==[True,True,True,False]
    active,side=support.conditions(f,"direction_flip");assert side[active].tolist()==[-1.0,1.0]


def test_causal_statistics_exclude_current_value():
    values=pd.Series(np.arange(61,dtype=float));z=support.causal_z(values,lookback=90,minimum=60)
    expected=(60-values.iloc[:60].mean())/values.iloc[:60].std(ddof=1);assert np.isclose(z.iloc[60],expected)
    rank=support.strict_prior_midrank(values,lookback=90,minimum=60);assert rank.iloc[60]==1.0
