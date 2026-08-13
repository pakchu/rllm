import numpy as np
import pandas as pd

from training import build_high_volatility_taker_imbalance_flow_persistence_support as support


def test_prior_rank_excludes_current() -> None:
    ranks=support.prior_rank(pd.Series(np.arange(181,dtype=float)))
    assert np.isnan(ranks.iloc[179]) and ranks.iloc[180]==1.0


def test_flow_features_capture_positive_serial_persistence_and_direction() -> None:
    imbalance=np.linspace(-0.4,0.6,96)
    quote=np.full(96,100.0)
    taker=quote*(imbalance+1)/2
    persistence,net=support.flow_features(quote,taker)
    assert persistence>.99 and net>0


def test_primary_onset_and_side_use_frozen_persistence() -> None:
    states=pd.DataFrame({"source_valid":[True]*4,"net_flow":[.01,.01,-.01,-.01],"variation_rank":[.6,.7,.8,.8],"persistence_rank":[.8,.8,.8,.7],"net_flow_magnitude_rank":[.2]*4})
    active,side=support.active(states,"primary")
    assert active.tolist()==[False,True,False,False]
    assert side.tolist()==[1,1,-1,-1]


def test_contract_is_frozen() -> None:
    assert support.PREREG_SHA=="ce94bf435f3939edc867deea1bb7757756d716f57ff968e677b89b56d39dbc29"
    assert support.CONTROLS==("no_persistence_gate","no_variation_gate","imbalance_level_tail","one_decision_stale_persistence","direction_flip","forced_long")
    assert "quote_asset_volume,taker_buy_quote" in support.QUERY
