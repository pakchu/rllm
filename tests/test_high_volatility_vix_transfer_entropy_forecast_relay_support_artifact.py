import json

from training import build_high_volatility_vix_transfer_entropy_forecast_relay_support as support

EXPECTED_CLOCK_SHA = "cb8c38ac92391724a2ed29512ce8632c65aa29cfafb239013590b8902ca8a899"
EXPECTED_FEATURE_SHA = "4d9eaedd0afecb7378541bd94f84ab0c1f662270b9ed1248b9b10ed6626cf306"
EXPECTED_RESULT_SHA = "757e1d7f16e1d1454c6370b5e33da92242d186dd6533ee0e535e9e99cfd18b08"


def test_reproduced_source_rejection_is_terminal_and_sealed() -> None:
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
    assert value["support"]["train"]["events"] == 1
    assert value["support"]["test"]["minority_side_share"] == 0.0
