import numpy as np
import pandas as pd
from training import build_high_volatility_intraday_variance_dispersion_relay_support as support


def test_prior_rank_excludes_current_value():
    ranks=support.prior_rank(pd.Series(list(range(181)),dtype=float))
    assert np.isnan(ranks.iloc[179]); assert ranks.iloc[180] == 1.0


def test_variance_dispersion_is_scale_free_and_subwindow_sensitive():
    uniform=np.full(480,.001); clustered=uniform.copy();clustered[:30]=.004
    total_u,disp_u=support.variance_dispersion(uniform)
    total_c,disp_c=support.variance_dispersion(clustered)
    assert total_c > total_u; assert disp_u == 0; assert disp_c > disp_u
    _,scaled=support.variance_dispersion(clustered*7)
    assert abs(scaled-disp_c) < 1e-12


def test_primary_onset_and_side_use_frozen_geometry():
    states=pd.DataFrame({'source_valid':[True]*4,'block_return':[.01,.01,-.01,-.01],'variation_rank':[.6,.7,.8,.8],'dispersion_rank':[.8,.8,.8,.7],'variance_dispersion':[1.,1.,1.,1.]})
    active,side=support.active(states,'primary')
    assert active.tolist()==[False,True,False,False]
    assert side.tolist()==[1,1,-1,-1]


def test_contract_is_frozen():
    assert support.PREREG_SHA == '7193b15406e0d88266e61fe81131993350e17a2b7a1ee448b20b56f1abd23328'
    assert support.CONTROLS == ('no_dispersion_tail','no_variation_gate','one_block_stale_geometry','direction_flip','forced_long')
    assert 'FROM bars_binance' in support.QUERY
