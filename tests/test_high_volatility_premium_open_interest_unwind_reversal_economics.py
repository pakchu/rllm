from training import evaluate_high_volatility_premium_open_interest_unwind_reversal_economics as economics
import pandas as pd

def test_frozen_accounting_and_stage_order_are_exact():
    assert economics.LEVERAGE==.5
    assert economics.BASE_COST==.0006
    assert economics.STRESS_COST==.001
    assert tuple(economics.STAGES)==("train","test","eval","final")
    assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"}
    assert economics.CONTROLS==("no_dvol_gate","no_premium_tail","no_oi_contraction","one_block_stale_features","direction_flip","forced_long")

def test_public_metrics_removes_trade_rows_only():
    assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}

def test_load_clock_allow_empty_preserves_zero_row_control(tmp_path):
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(path, index=False, compression="gzip")
    value = economics.load_clock_allow_empty(path, "train", pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    assert value.empty
    assert str(value.entry_time.dtype) == "datetime64[ns, UTC]"
