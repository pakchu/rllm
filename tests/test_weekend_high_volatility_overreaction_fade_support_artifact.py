import json

from training import build_weekend_high_volatility_overreaction_fade_support as support


def test_whvof_support_is_terminal_before_novelty_and_outcomes():
    report = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert report["support"]["train"]["events"] == 0
    assert report["support"]["test"]["events"] == 9
    assert report["support"]["eval"]["events"] == 7
    assert report["support"]["final"]["events"] == 11
    assert report["decision"] == "terminal_source_support_reject"
