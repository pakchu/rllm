from training import evaluate_high_volatility_macro_polarity_concordance_relay_economics as economics
def test_frozen_accounting_stage_order_and_controls():
 assert economics.POLICY_ID=="HVMPC-24";assert economics.LEVERAGE==.5;assert economics.BASE_COST==.0006;assert economics.STRESS_COST==.001;assert tuple(economics.STAGES)==("train","test","eval","final");assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert economics.CONTROLS==("news_only","epu_only","no_variation_gate","one_week_stale_pair","direction_flip","same_clock_forced_long")
def test_public_metrics_removes_trade_rows_only():assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}
