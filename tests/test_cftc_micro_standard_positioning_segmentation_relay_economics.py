from training import evaluate_cftc_micro_standard_positioning_segmentation_relay_economics as economics
import pandas as pd

def test_frozen_accounting_and_stage_order_are_exact():
 assert economics.LEVERAGE==.5 and economics.BASE_COST==.0006 and economics.STRESS_COST==.001;assert tuple(economics.STAGES)==("train","test","eval","final");assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert economics.CONTROLS==("no_variation_gate","standard_asset_manager_only","micro_leveraged_only","one_report_stale_segmentation","direction_flip","forced_long")
def test_public_metrics_removes_trade_rows_only():assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}
def test_load_clock_allow_empty_preserves_zero_row_control(tmp_path):
 p=tmp_path/"empty.csv.gz";pd.DataFrame(columns=["entry_time","exit_time","side"]).to_csv(p,index=False,compression="gzip");v=economics.load_clock_allow_empty(p,"train",pd.Timestamp("2023-07-01",tz="UTC"),pd.Timestamp("2024-01-01",tz="UTC"));assert v.empty and str(v.entry_time.dtype)=="datetime64[ns, UTC]"
