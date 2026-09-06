from training import evaluate_high_volatility_meme_attention_saturation_reversal_economics as economics
def test_frozen_accounting_stage_order_and_controls():
 assert economics.LEVERAGE==.5;assert economics.BASE_COST==.0006;assert economics.STRESS_COST==.001;assert tuple(economics.STAGES)==("train","test","eval","final");assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert economics.CONTROLS==("no_doge_return_tail","no_doge_turnover_tail","no_btc_variation_gate","one_block_stale_doge","direction_flip","same_clock_forced_long")
def test_public_metrics_removes_trade_rows_only():assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}
