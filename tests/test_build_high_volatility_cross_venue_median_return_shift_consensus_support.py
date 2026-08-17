import pandas as pd
from training import build_high_volatility_cross_venue_median_return_shift_consensus_support as s
def primary(side:int=1)->pd.DataFrame:
 d=pd.Timestamp("2023-07-01T01:00Z");e=d+pd.Timedelta("5m");return pd.DataFrame({"candidate":["HVMRSR-8"],"control":["primary"],"split":["train"],"decision_time":[d],"feature_available_time":[d],"entry_time":[e],"exit_time":[e+pd.Timedelta("8h")],"side":[side],"median_shift":[side*.1],"shift_rank":[.8],"variation_rank":[.8]})
def spot(shift:float)->pd.DataFrame:return pd.DataFrame({"decision_time":[pd.Timestamp("2023-07-01T01:00Z")],"source_valid":[True],"spot_shift":[shift]})
def test_same_sign_spot_shift_confirms_primary():
 out=s.confirm(primary(1),spot(.2));assert len(out)==1 and out.iloc[0]["side"]==1
def test_opposite_spot_shift_rejects_primary():assert s.confirm(primary(1),spot(-.2)).empty
def test_prepare_spot_validates_exact_block():
 d=pd.Timestamp("2023-07-01T01:00Z");start=d-pd.Timedelta("8h");raw=pd.DataFrame({"decision_time":[d],"first_median":[-.1],"second_median":[.1],"source_rows":[480],"distinct_rows":[480],"first_ts":[start],"last_ts":[d-pd.Timedelta("1m")],"coherent":[True]})
 assert s.prepare_spot(raw).iloc[0]["source_valid"]
