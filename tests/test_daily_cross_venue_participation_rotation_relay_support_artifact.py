import json


def test_dcvpr_support_artifact_is_terminal_before_outcomes():
    result = json.load(open("results/daily_cross_venue_participation_rotation_relay_support_2026-08-08.json"))
    assert result["decision"] == "terminal_source_support_reject"
    assert result["support_passed"] is False
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert all(stage["events"] == 0 for stage in result["support"].values())
