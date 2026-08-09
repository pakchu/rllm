from training import evaluate_high_volatility_dvol_variation_risk_relay_economics as economics

def test_frozen_accounting_and_stage_order_are_exact():
    assert economics.LEVERAGE==.5
    assert economics.BASE_COST==.0006
    assert economics.STRESS_COST==.001
    assert tuple(economics.STAGES)==("train","test","eval","final")
    assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"}
    assert economics.CONTROLS==("no_dvol_variation_gate","no_btc_variation_gate","dvol_direction","one_day_stale_dvol","direction_flip","same_clock_forced_long")

def test_public_metrics_removes_trade_rows_only():
    assert economics.public_metrics({"x":1,"trade_rows":[1]})=={"x":1}
