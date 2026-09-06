import json
from pathlib import Path


def test_source_support_artifact_passes_with_frozen_counts():
    value = json.loads(Path("results/high_volatility_premium_open_interest_unwind_reversal_support_2026-08-11.json").read_text())
    assert value["policy_id"] == "HVPOIUR-8"
    assert value["support_passed"] is True
    assert value["decision"] == "pass_to_novelty"
    assert {name: item["events"] for name, item in value["support"].items()} == {"train": 58, "test": 88, "eval": 77, "final": 54}
    assert value["postentry_return_pnl_execution_price_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert all(item["promotion_authorized"] is False for item in value["controls"].values())
