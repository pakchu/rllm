from training import evaluate_high_volatility_low_frequency_energy_relay_economics as economics
def test_frozen_accounting_stage_order_and_controls():
 assert economics.LEVERAGE==.5;assert economics.BASE_COST==.0006;assert economics.STRESS_COST==.001;assert tuple(economics.STAGES)==("train","test","eval","final");assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert economics.CONTROLS==("no_low_frequency_tail","no_variation_gate","raw_low_frequency_share_nonnegative","one_block_stale_geometry","direction_flip","forced_long")
def test_public_metrics_removes_trade_rows_only():assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}
