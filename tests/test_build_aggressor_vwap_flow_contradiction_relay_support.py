import pandas as pd
from training import build_aggressor_vwap_flow_contradiction_relay_support as support

def _row(**kw):
 d={"source_day":"2024-07-01","first_open":100.,"last_close":101.,"total_base":100.,"total_quote":10000.,"buy_base":40.,"buy_quote":4040.,"source_rows":360,"distinct_rows":360,"first_ts":"2024-07-01T18:00Z","last_ts":"2024-07-01T23:59Z","coherent":True};d.update(kw);return pd.DataFrame([d])

def test_buy_sell_vwap_contradiction_and_side():
 f=support.build_features(_row())
 assert f.iloc[0].buy_vwap>f.iloc[0].sell_vwap and f.iloc[0].signed_taker_flow<0 and f.iloc[0].flow_contradiction
 assert support.signal(f,"primary").iloc[0]==1 and support.signal(f,"direction_flip").iloc[0]==-1

def test_incomplete_or_invalid_residual_source_fails_closed():
 assert not support.build_features(_row(source_rows=359)).iloc[0].source_valid
 assert not support.build_features(_row(buy_base=101.)).iloc[0].source_valid

def test_clock_and_frozen_gates():
 f=support.build_features(_row());c=support.build_clock(f)
 assert c.iloc[0].entry_time==pd.Timestamp("2024-07-02T00:05Z") and c.iloc[0].exit_time==pd.Timestamp("2024-07-02T12:05Z")
 assert support.MINIMUM=={"train":8,"test":12,"eval":12,"final":8}

def test_source_evaluator_keeps_outcomes_closed():
 s=open(support.__file__).read();assert '"execution_prices_opened":False' in s;assert '"postentry_return_or_pnl_opened":False' in s;assert '"gross9_rows_opened":False' in s
