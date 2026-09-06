import json

from training import build_high_volatility_quarter_hour_public_signal_relay_support as support


EXPECTED_CLOCK_SHA = "2200d3813a18be5e703c9e94367f266d7862a0625a9c321574d2f9f4ac2bea49"
EXPECTED_FEATURE_SHA = "f2064be47ce2a455cec61c9035d2d6e9c9122bcff8213a3e9f51671ae53dc2be"
EXPECTED_RESULT_SHA = "5036b79f70801594bfa216b12cd4a0aba8c258c6bcb2ae7ef3ca0b000da8eb5a"


def test_reproduced_source_failure_is_terminal_and_sealed() -> None:
    value = json.loads(support.RESULT.read_text())
    assert support.sha256(support.CLOCK) == EXPECTED_CLOCK_SHA
    assert support.sha256(support.FEATURES) == EXPECTED_FEATURE_SHA
    assert support.sha256(support.RESULT) == EXPECTED_RESULT_SHA
    assert value["support_passed"] is False
    assert value["advance_to_gross9_novelty"] is False
    assert value["advance_to_economic_outcomes"] is False
    assert value["gross9_rows_opened"] is False
    assert value["postentry_return_pnl_execution_price_opened"] is False
    assert value["decision"] == "terminal_source_support_reject"
    assert value["support"]["train"]["minority_side_share"] < 0.20
    assert value["support"]["train"]["max_month_share"] > 0.45
    assert value["support"]["final"]["max_month_share"] > 0.45
