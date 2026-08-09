import numpy as np,pandas as pd
from training import build_high_volatility_har_variation_surprise_relay_support as s
def test_rank_is_current_excluded(monkeypatch):
 monkeypatch.setattr(s,'rank',s.rank);x=pd.Series(list(range(181)),dtype=float);r=s.rank(x);assert np.isnan(r.iloc[179]);assert r.iloc[180]==1.
def test_contract():assert s.PREREG_SHA=='a6a6f386218949d9e5d37d3168738970fea38cfd0ed09be5c0a7f0b0fc9ce44f';assert 'FROM bars_binance' in s.QUERY;assert s.CONTROLS==('no_surprise_tail','no_efficiency_gate','raw_positive_surprise','one_block_stale_geometry','direction_flip','forced_long')
