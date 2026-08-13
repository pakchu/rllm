import numpy as np
import pandas as pd
from training import build_high_volatility_bds_nonlinear_dependence_relay_support as s


def test_prior_rank_excludes_current_value():
    ranks=s.prior_rank(pd.Series(list(range(181)),dtype=float))
    assert np.isnan(ranks.iloc[179]); assert ranks.iloc[180] == 1.0


def test_bds_departure_detects_repeated_two_step_histories():
    iid=np.random.default_rng(1).normal(size=96)
    repeated=np.repeat(np.array([-1.,1.]),48)
    iid_departure,_,_=s.bds_departure(iid)
    repeated_departure,c1,c2=s.bds_departure(repeated)
    assert np.isfinite(iid_departure)
    assert repeated_departure > iid_departure
    assert 0 <= c1 <= 1 and 0 <= c2 <= 1


def test_primary_onset_and_side_use_frozen_dependence():
    states=pd.DataFrame({
        'source_valid':[True]*4,'direction_confirmed':[True]*4,
        'block_return':[.01,.01,-.01,-.01],
        'variation_rank':[.6,.7,.8,.8],
        'departure_rank':[.8,.8,.8,.7],
        'bds_departure':[.1]*4,'lag1_rank':[.2]*4,
    })
    active,side=s.active(states,'primary')
    assert active.tolist()==[False,True,False,False]
    assert side.tolist()==[1,1,-1,-1]


def test_contract_is_frozen():
    assert s.PREREG_SHA == '9e10c67e2f1c589e3467abafa53b77f85842e9134dfddcdca6fec831aa2977cc'
    assert s.CONTROLS == ('no_departure_tail','no_variation_gate','linear_lag1_autocorrelation_tail','one_decision_stale_dependence','direction_flip','forced_long')
    assert 'FROM bars_binance' in s.QUERY
