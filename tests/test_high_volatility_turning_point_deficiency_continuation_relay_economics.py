import pandas as pd
from training import evaluate_high_volatility_turning_point_deficiency_continuation_relay_economics as economics
def test_frozen_accounting_stage_order_and_controls():
 assert economics.LEVERAGE==.5;assert economics.BASE_COST==.0006;assert economics.STRESS_COST==.001;assert tuple(economics.STAGES)==("train","test","eval","final");assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert economics.CONTROLS==("no_turning_point_deficiency_tail","no_variation_gate","turning_point_share_below_two_thirds","one_block_stale_geometry","direction_flip","forced_long")
def test_public_metrics_removes_trade_rows_only():assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}


def test_load_clock_allow_empty_preserves_zero_row_control(tmp_path):
 path=tmp_path/"empty.csv.gz"
 pd.DataFrame(columns=["entry_time","exit_time","side"]).to_csv(path,index=False,compression="gzip")
 clock=economics.load_clock_allow_empty(path,"train",pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z"))
 assert clock.empty
 assert str(clock.entry_time.dtype)=="datetime64[ns, UTC]"
