from training.symbolic_action_ridge import target_value


def test_target_value_reads_reward_fallback_for_rex_ranker_rows():
    row = {"reward": {"net_return": 0.02, "mae": 0.005, "mfe": 0.03, "utility": 0.015}, "action": {"hold_bars": 144, "family": "rex", "side": "LONG"}}
    assert target_value(row, target="net_return") == 0.02
    assert target_value(row, target="risk_adjusted") == 0.015
    assert target_value(row, target="utility") == 0.015
