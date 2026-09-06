import json
from pathlib import Path


def test_source_rejection_is_terminal_before_outcomes():
    value=json.loads(Path("results/funding_cash_transfer_basis_inventory_reanchoring_reversal_support_2026-08-11.json").read_text())
    assert value["policy_id"]=="FCBIRR-8"
    assert value["support_passed"] is False and value["decision"]=="terminal_source_support_reject"
    assert value["advance_to_gross9_novelty"] is False and value["advance_to_economic_outcomes"] is False
    assert {k:v["events"] for k,v in value["support"].items()}=={"train":49,"test":84,"eval":18,"final":41}
    assert value["support"]["eval"]["shorts"]==0 and value["support"]["final"]["shorts"]==0
    assert value["postentry_return_pnl_execution_price_opened"] is False and value["gross9_rows_opened"] is False
