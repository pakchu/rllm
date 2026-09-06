import numpy as np
import pandas as pd
from training import build_high_volatility_intrabar_acceptance_breadth_relay_support as support

def test_prior_rank_excludes_current_value():
 v=pd.Series(list(range(181)),dtype=float);r=support.prior_rank(v);assert np.isnan(r.iloc[179]);assert r.iloc[180]==1.

def test_acceptance_breadth_counts_constituent_close_locations():
 w=pd.DataFrame({"open":np.ones(480),"high":np.full(480,2.),"low":np.zeros(480),"close":np.r_[np.full(360,1.5),np.full(120,.5)]});assert support.acceptance_breadth(w)==.5

def test_contract_is_frozen():
 assert support.PREREG_SHA=="619674be60397f0004ad79b089c8599db59553fbdbd3499cdbc9e3fca502c7ef";assert "FROM bars_binance" in support.QUERY;assert support.CONTROLS==("no_acceptance_tail","no_variation_gate","raw_absolute_acceptance_above_quarter","one_block_stale_acceptance","direction_flip","forced_long")
