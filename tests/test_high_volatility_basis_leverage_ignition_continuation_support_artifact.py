import json
from pathlib import Path


def test_source_support_passes_frozen_gates():
    value=json.loads(Path("results/high_volatility_basis_leverage_ignition_continuation_support_2026-08-11.json").read_text())
    assert value["policy_id"]=="HVBLIC-6" and value["support_passed"] is True
    assert {k:v["events"] for k,v in value["support"].items()}=={"train":122,"test":204,"eval":184,"final":102}
    assert value["postentry_return_pnl_execution_price_opened"] is False and value["gross9_rows_opened"] is False
    assert all(not x["promotion_authorized"] for x in value["controls"].values())
