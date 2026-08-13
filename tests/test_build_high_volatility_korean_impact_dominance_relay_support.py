import numpy as np,pandas as pd
from training import build_high_volatility_korean_impact_dominance_relay_support as s
def test_prior_statistics_exclude_current():
 x=pd.Series(range(181),dtype=float);r=s.prior_rank(x);z=s.prior_zscore(x);assert r.iloc[:180].isna().all() and z.iloc[:180].isna().all();assert r.iloc[180]==1.;assert z.iloc[180]==(180-np.mean(np.arange(180)))/np.std(np.arange(180),ddof=1)
def test_onset_side_controls_and_hash():
 d=pd.date_range("2024-07-01T04:00:00Z",periods=4,freq="8h");x=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"upbit_impact":[1.]*4,"binance_impact":[0.]*4,"upbit_impact_z":[1.]*4,"binance_impact_z":[0.]*4,"korean_impact_dominance":[1.]*4,"dominance_rank":[.7,.8,.9,.7],"raw_impact_difference":[1.]*4,"raw_difference_rank":[.8]*4,"binance_realized_variation":[.01]*4,"variation_rank":[.7]*4,"upbit_block_return":[.01]*4,"side":[1]*4,"eligible":[False,True,True,False]},columns=s.PANEL_COLS);onset,side=s.active(x);assert onset.tolist()==[False,True,False,False] and side.tolist()==[1]*4;assert s.active(x,"direction_flip")[1].tolist()==[-1]*4;assert s.PREREG_SHA=="f7c369f980a8c788a585a17dc5409b46005188b4a38e0743920c33ae748ac5c3"
def test_query_is_source_only_and_uses_common_base_volume():
 q=" ".join(s.QUERY.lower().split());assert "bars_upbit" in q and "bars_binance" in q and "sum(volume)" in q;assert all(x not in q for x in ("funding","pnl","gross9"))
